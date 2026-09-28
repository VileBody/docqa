"""Offline inbox for R originals. Never downloads, invents gold or replaces snapshots.

Identity markers are a screening heuristic, not proof of provenance. Review the
PDF visually against acquisition/cards before importing; origin is user-attested.
"""
from __future__ import annotations

from pathlib import Path
import re
import unicodedata
from typing import Any, Iterable

from .core import Corpus
from .public import import_public_file, review_template
from .util import read_json, sha256

MAX_BYTES = 30_000_000


def _search_key(value: str) -> str:
    """Ignore layout punctuation; do NOT conflate Latin and Cyrillic characters."""
    value = unicodedata.normalize("NFKC", value).casefold()
    return "".join(ch for ch in value if ch.isalnum())


class PublicAcquisition:
    """Inspect incoming/R/R01.pdf ... R08.pdf and register immutable originals.

    No GPU, LLM or network is used. PDF inspection/import needs the ``pdf`` extra.
    The source registry lives next to the corpus in acquisition/registry.json.
    """

    def __init__(self, corpus: Corpus | str | Path | None = None, *, registry: str | Path | None = None):
        self.corpus = corpus if isinstance(corpus, Corpus) else Corpus(corpus)
        self.project = self.corpus.root.parent
        path = Path(registry) if registry is not None else self.project / "acquisition/registry.json"
        payload = read_json(path)
        records = payload["sources"]
        self.sources = {row["id"]: row for row in records}
        if len(self.sources) != len(records):
            raise ValueError("Duplicate IDs in acquisition registry")
        for doc_id, row in self.sources.items():
            d = self.corpus.get(doc_id)
            if d.metadata["suite"] != "R" or row["expected_pages"] != d.metadata["pages"]:
                raise ValueError(f"{doc_id}: acquisition registry disagrees with corpus manifest")
            if row["filename"] != f"{doc_id}.pdf":
                raise ValueError(f"{doc_id}: expected an exact ID.pdf inbox filename")

    def _ids(self, ids: Iterable[str] | None) -> list[str]:
        selected = list(ids) if ids is not None else sorted(self.sources)
        if not selected or len(set(selected)) != len(selected):
            raise ValueError("Select at least one public ID, without duplicates")
        unknown = set(selected) - self.sources.keys()
        if unknown:
            raise ValueError(f"Unknown public IDs: {sorted(unknown)}")
        return selected

    def status(self, directory: str | Path | None = None) -> dict[str, Any]:
        inbox = Path(directory) if directory is not None else self.project / "incoming/R"
        rows = []
        for doc_id in self._ids(None):
            d = self.corpus.get(doc_id)
            file = inbox / self.sources[doc_id]["filename"]
            expected = d.metadata.get("sha256")
            registered = bool(d.available and expected and sha256(d.pdf_path) == expected)
            bad = bool(d.available and (not expected or not registered))
            questions = d.questions(ready_only=False)
            approved = [q for q in questions if q["status"].startswith("ready")]
            independently_reviewed = sum(q.get("review", {}).get("independent_human_review") is True for q in approved)
            stage = ("snapshot_error" if bad else
                     "reviewed" if registered and questions and independently_reviewed == len(questions) else
                     "downloaded_needs_review" if registered else
                     "in_inbox_unchecked" if file.is_file() else "missing")
            rows.append({"id": doc_id, "stage": stage, "inbox_file": str(file),
                         "inbox_present": file.is_file(), "registered": registered,
                         "expected_pages": self.sources[doc_id]["expected_pages"],
                         "questions": len(questions), "approved_questions": len(approved),
                         "independently_reviewed_questions": independently_reviewed,
                         "source_url": self.sources[doc_id]["url"]})
        return {"ok": not any(r["stage"] == "snapshot_error" for r in rows),
                "registered_originals": sum(r["registered"] for r in rows),
                "inbox_present": sum(r["inbox_present"] for r in rows),
                "required_originals": len(rows),
                "all_downloaded": all(r["registered"] for r in rows),
                "public_benchmark_ready": all(r["stage"] == "reviewed" for r in rows),
                "results": rows,
                "note": "Status is not certification of legal currency, origin or semantic correctness."}

    def inspect_file(self, document_id: str, path: str | Path) -> dict[str, Any]:
        self._ids([document_id])
        from pypdf import PdfReader
        spec = self.sources[document_id]
        file = Path(path).expanduser().resolve()
        result: dict[str, Any] = {"id": document_id, "path": str(file), "ok": False,
                                 "errors": [], "warnings": [], "provenance_verified": False}
        errors = result["errors"]
        if not file.is_file():
            errors.append("missing_file")
            return result
        size = file.stat().st_size
        result["bytes"] = size
        if size == 0 or size > MAX_BYTES:
            errors.append("empty_or_oversized_file")
            return result
        with file.open("rb") as f:
            if not f.read(8).startswith(b"%PDF-"):
                errors.append("not_pdf_bytes (possibly an HTML error page)")
                return result
        result["sha256"] = sha256(file)
        try:
            reader = PdfReader(file, strict=False)
            if reader.is_encrypted:
                errors.append("encrypted_pdf")
                return result
            count = len(reader.pages)
            result["pages"] = count
            if count != spec["expected_pages"]:
                errors.append(f"page_count_mismatch: expected {spec['expected_pages']}, got {count}")
                return result
            texts = [page.extract_text() or "" for page in reader.pages]
        except Exception as exc:
            errors.append(f"pdf_parse_failed: {type(exc).__name__}: {exc}")
            return result
        result["characters_per_page"] = [len(t) for t in texts]
        key = _search_key("\n".join(texts))
        if len(key) < 40:
            errors.append("no_usable_text_layer; preserve the original, do not silently add OCR")
        groups = []
        for group in spec["identity_groups"]:
            hits = [term for term in group["any_of"] if _search_key(term) in key]
            groups.append({"name": group["name"], "matched": bool(hits), "hits": hits})
            if not hits:
                errors.append(f"identity_marker_missing: {group['name']}; visually review the file/extraction")
        result["identity_groups"] = groups
        if any(len(t.strip()) < 40 for t in texts):
            result["warnings"].append("Some pages have little extracted text; compare them to the PDF before annotation.")
        result["warnings"].append("Markers and page count cannot prove origin or complete identity. Check title/date/version visually.")
        result["ok"] = not errors
        return result

    def import_directory(self, directory: str | Path | None = None, *, ids: Iterable[str] | None = None,
                         reviews: str | Path | None = None, require_all: bool = False) -> dict[str, Any]:
        inbox = Path(directory) if directory is not None else self.project / "incoming/R"
        review_dir = Path(reviews) if reviews is not None else self.project / "reviews/R"
        selected = self._ids(ids)
        results: list[dict[str, Any]] = []
        for doc_id in selected:
            d = self.corpus.get(doc_id)
            file = inbox / self.sources[doc_id]["filename"]
            try:
                expected = d.metadata.get("sha256")
                if d.available and expected:
                    if sha256(d.pdf_path) != expected:
                        raise ValueError("Registered PDF hash mismatch; no replacement attempted")
                    if file.is_file() and sha256(file) != expected:
                        raise ValueError("Inbox differs from immutable registered snapshot; no replacement attempted")
                    row = {"id": doc_id, "ok": True, "status": "already_registered", "sha256": expected}
                elif not file.is_file():
                    results.append({"id": doc_id, "ok": not require_all, "status": "missing", "path": str(file)})
                    continue
                else:
                    inspection = self.inspect_file(doc_id, file)
                    if not inspection["ok"]:
                        results.append({"id": doc_id, "ok": False, "status": "rejected", "inspection": inspection})
                        continue
                    row = import_public_file(self.corpus, doc_id, file)
                    row["inspection"] = inspection
                template = review_dir / f"{doc_id}.json"
                if template.exists():
                    prior = read_json(template)
                    if prior.get("document_id") != doc_id or prior.get("pdf_sha256") != row["sha256"]:
                        raise ValueError("Existing review template is stale or belongs to another document; not overwritten")
                    row["review_template_status"] = "existing_preserved"
                else:
                    review_template(self.corpus, doc_id, template)
                    row["review_template_status"] = "created_needs_review"
                row["review_template"] = str(template)
                results.append(row)
            except (ValueError, TypeError, OSError, KeyError) as exc:
                results.append({"id": doc_id, "ok": False, "status": "error", "error": f"{type(exc).__name__}: {exc}"})
        recognized = {self.sources[i]["filename"] for i in self.sources}
        extras = sorted(p.name for p in inbox.glob("*") if p.is_file() and p.suffix.lower() == ".pdf" and p.name not in recognized)
        return {"ok": all(r["ok"] for r in results), "selected": len(selected),
                "missing": [r["id"] for r in results if r["status"] == "missing"],
                "ignored_pdf_filenames": extras, "results": results,
                "public_benchmark_ready": self.status(inbox)["public_benchmark_ready"],
                "note": "Import freezes bytes and creates review templates. It does not approve draft gold. Inbox files are retained."}
