"""Render explicitly paginated source ASTs. Fonts are discovered, never bundled.

Canonical coordinates reference the authored UTF-8 text, not a particular PDF
parser. Every rendered block receives a physical-page/bbox mapping. Page titles
are included; running headers, footers and decorative elements are excluded.
"""
from __future__ import annotations
import os
from pathlib import Path
from xml.sax.saxutils import escape
from typing import Any
from .util import PAGE_SEPARATOR, text_hash, write_json, write_text


def find_fonts(font_path: str | Path | None = None, bold_path: str | Path | None = None) -> tuple[Path, Path]:
    regular = font_path or os.getenv("DOCQA_FONT")
    bold = bold_path or os.getenv("DOCQA_FONT_BOLD")
    pairs = [
        ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        ("/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf", "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"),
        ("/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
        ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/arialbd.ttf"),
    ]
    if regular:
        a, b = Path(regular).expanduser(), Path(bold or regular).expanduser()
        if not a.is_file() or not b.is_file():
            raise FileNotFoundError("Font file not found. Set DOCQA_FONT and optionally DOCQA_FONT_BOLD.")
        return a, b
    for a, b in pairs:
        if Path(a).is_file() and Path(b).is_file():
            return Path(a), Path(b)
    raise FileNotFoundError("No Cyrillic TrueType font found. Pass font_path=... or set DOCQA_FONT. Font files are not bundled.")


def render_document(source: dict[str, Any], pdf_path: Path, sidecar_dir: Path,
                    font_path: str | Path | None = None, bold_path: str | Path | None = None) -> dict:
    from reportlab.pdfgen.canvas import Canvas
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.colors import HexColor
    from reportlab.platypus import Paragraph, Table, TableStyle
    from reportlab.lib.pagesizes import A4

    regular, bold = find_fonts(font_path, bold_path)
    # A distinct name allows different font pairs in one Python process.
    key = text_hash(str(regular.resolve()) + str(bold.resolve()))[:10]
    normal_name, bold_name = f"DocQA-{key}", f"DocQA-{key}-Bold"
    for name, file in [(normal_name, regular), (bold_name, bold)]:
        if name not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(TTFont(name, str(file)))
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar_dir.mkdir(parents=True, exist_ok=True)
    width, height = A4
    margin = float(source.get("layout", {}).get("margin", 48))
    body_size = float(source.get("layout", {}).get("font_size", 10.4))
    if not 8 <= body_size <= 13:
        raise ValueError("Body font size must be between 8 and 13 pt")
    body = ParagraphStyle("body", fontName=normal_name, fontSize=body_size,
                          leading=body_size * 1.45, textColor=HexColor("#222B34"),
                          alignment=TA_LEFT, splitLongWords=True)
    heading = ParagraphStyle("heading", parent=body, fontName=bold_name, fontSize=15.5, leading=20)
    subheading = ParagraphStyle("subheading", parent=body, fontName=bold_name, fontSize=11.3, leading=16)
    small = ParagraphStyle("small", parent=body, fontSize=8.5, leading=12)
    cell = ParagraphStyle("cell", parent=body, fontSize=7.5, leading=10.2)
    c = Canvas(str(pdf_path), pagesize=A4, pageCompression=1, invariant=1)
    c.setTitle(source["title"])
    c.setAuthor("DocQA Corpus - synthetic educational document")
    c.setSubject("Fictional document. Not an official publication; no legal advice.")
    page_texts: list[str] = []
    locations: dict[str, dict] = {}
    page_map = []
    global_start = 0
    n = len(source["pages"])

    for page_no, page in enumerate(source["pages"], 1):
        c.setFillColor(HexColor("#536171"))
        c.setFont(normal_name, 7)
        c.drawString(margin, height - 25, "УЧЕБНЫЙ ДОКУМЕНТ. ОРГАНИЗАЦИИ И ОБСТОЯТЕЛЬСТВА ВЫМЫШЛЕНЫ")
        c.setStrokeColor(HexColor("#C4CCD4"))
        c.line(margin, height - 33, width - margin, height - 33)
        y = height - 54
        text_parts: list[str] = []
        local_len = 0

        def locate(block_id: str, text: str, bbox: list[float], section: str | None, kind: str) -> None:
            nonlocal local_len
            if block_id in locations:
                raise ValueError(f"Duplicate block ID: {block_id}")
            if text_parts:
                text_parts.append("\n\n")
                local_len += 2
            start = global_start + local_len
            text_parts.append(text)
            local_len += len(text)
            locations[block_id] = {"block_id": block_id, "page": page_no,
                "printed_page": page.get("printed_page"), "section": section,
                "char_start": start, "char_end": start + len(text), "text": text,
                "bbox": [round(v, 2) for v in bbox], "bbox_origin": "top-left", "kind": kind}

        title = page["title"]
        p = Paragraph(escape(title), heading)
        _, h = p.wrap(width - 2 * margin, height)
        p.drawOn(c, margin, y - h)
        locate(f"page-{page_no}-title", title, [margin, height-y, width-margin, height-y+h], page.get("section", title), "heading")
        y -= h + 19
        for block in page["blocks"]:
            kind = block.get("kind", "paragraph")
            section = block.get("section", page.get("section", title))
            if kind in ("table_header", "table_row"):
                cells = block["cells"]
                col_widths = block.get("widths", [.20, .13, .23, .10, .09, .10, .15])
                if len(cells) != len(col_widths):
                    raise ValueError("Table cell/column count mismatch")
                widths = [(width - 2*margin)*v/sum(col_widths) for v in col_widths]
                tab = Table([[Paragraph(escape(str(v)), cell) for v in cells]], colWidths=widths)
                styles = [("GRID", (0,0), (-1,-1), .4, HexColor("#AAB7C5")),
                          ("VALIGN", (0,0), (-1,-1), "TOP"),
                          ("LEFTPADDING", (0,0), (-1,-1), 5),
                          ("RIGHTPADDING", (0,0), (-1,-1), 5),
                          ("TOPPADDING", (0,0), (-1,-1), 6),
                          ("BOTTOMPADDING", (0,0), (-1,-1), 6)]
                if kind == "table_header":
                    styles.append(("BACKGROUND", (0,0), (-1,-1), HexColor("#E9EFF4")))
                tab.setStyle(TableStyle(styles))
                _, bh = tab.wrap(width - 2*margin, height)
                if y - bh < 54:
                    raise ValueError(f"Page overflow: {source['id']} p{page_no} block {block['id']}")
                tab.drawOn(c, margin, y - bh)
                locate(block["id"], "\n".join(map(str,cells)), [margin, height-y,width-margin,height-y+bh], section, kind)
                y -= bh
            else:
                text = block["text"]
                style = subheading if kind == "subheading" else small if kind == "footnote" else body
                p = Paragraph(escape(text).replace("\n", "<br/>"), style)
                _, bh = p.wrap(width - 2*margin, height)
                if block.get("position") == "bottom":
                    y = min(y, 57 + bh)
                if y - bh < 49:
                    raise ValueError(f"Page overflow: {source['id']} p{page_no} block {block['id']} at y={y}, h={bh}")
                p.drawOn(c, margin, y - bh)
                locate(block["id"], text, [margin, height-y,width-margin,height-y+bh], section, kind)
                y -= bh + (5 if kind == "footnote" else 11)
        c.setStrokeColor(HexColor("#C4CCD4"))
        c.line(margin, 35, width-margin, 35)
        c.setFont(normal_name, 7.3)
        c.setFillColor(HexColor("#596675"))
        c.drawString(margin, 22, page.get("footer", "Документ сформирован автоматически"))
        label = page.get("printed_page")
        if label is None and not source.get("special_pagination"):
            label = str(page_no)
        if label is not None:
            c.drawRightString(width-margin, 22, str(label))
        text = "".join(text_parts)
        page_texts.append(text)
        page_map.append({"page":page_no,"printed_page":label,"char_start":global_start,
                         "char_end":global_start+len(text),"text":text})
        global_start += len(text) + (len(PAGE_SEPARATOR) if page_no < n else 0)
        c.showPage()
    c.save()
    canonical = PAGE_SEPARATOR.join(page_texts)
    write_text(sidecar_dir/"canonical.txt", canonical)
    write_json(sidecar_dir/"pages.json", page_map)
    write_json(sidecar_dir/"anchors.json", locations)
    write_json(sidecar_dir/"source.json", source)
    md = [f"# {source['title']}", "", "Учебная синтетика. Не является официальным документом.", ""]
    for page_no,page in enumerate(source['pages'],1):
        md += [f"<!-- physical-page: {page_no} -->", f"## {page['title']}", ""]
        for b in page['blocks']:
            md += [" | ".join(map(str,b['cells'])) if 'cells' in b else b['text'], ""]
    write_text(sidecar_dir/"source.md", "\n".join(md))
    return {"pages": n, "characters":len(canonical), "whitespace_words":len(canonical.split()),
            "canonical_sha256":text_hash(canonical),"anchors":locations,
            "font_family_used":regular.stem,"coordinate_system":"authored_source_v1"}
