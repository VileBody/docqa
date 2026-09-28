"""Small stdlib-only helpers; JSON writes are atomic within one filesystem."""
from __future__ import annotations
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Iterable

PAGE_SEPARATOR = "\n\f\n"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    write_text(path, json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    write_text(path, "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def safe_id(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{1,63}", value):
        raise ValueError("ID must be 2-64 ASCII letters, digits, '_' or '-', starting with a letter")
    return value


def compact(text: str) -> str:
    """Whitespace-only normalization for checking text-layer presence, not meaning."""
    return "".join(text.replace("\u00ad", "").split())
