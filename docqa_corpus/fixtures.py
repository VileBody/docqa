"""Small, non-malicious file fixtures and a deliberately conservative preflight.

Preflight is a reference *policy*, not a security scanner or OCR implementation.
It rejects image-only body pages, while allowing genuinely blank separator pages.
PDF parsing of untrusted uploads belongs in a resource-limited worker in production.
"""
from __future__ import annotations
from io import BytesIO
from pathlib import Path
from typing import Any
from .core import Corpus
from .util import read_json, write_json, sha256


def preflight(path: str | Path, *, max_pages: int = 100, max_bytes: int = 30_000_000) -> dict[str, Any]:
    from pypdf import PdfReader
    p = Path(path)
    if max_pages < 1 or max_bytes < 1:
        raise ValueError('Limits must be positive')
    if not p.is_file():
        return {'ok': False, 'code': 'file_not_found'}
    if p.stat().st_size > max_bytes:
        return {'ok': False, 'code': 'file_too_large'}
    with p.open('rb') as stream:
        if not stream.read(8).startswith(b'%PDF-'):
            return {'ok': False, 'code': 'unsupported_media_type'}
    try:
        reader = PdfReader(p, strict=True)
        if reader.is_encrypted:
            return {'ok': False, 'code': 'password_required'}
        count = len(reader.pages)
        if count > max_pages:
            return {'ok': False, 'code': 'too_many_pages', 'pages': count}
        texts = [(page.extract_text() or '').strip() for page in reader.pages]
        # Image-only body pages in our fixtures have no text at all. Real-world
        # PDFs with text headers and scanned body require stronger layout checks.
        image_only = [i + 1 for i, (page, text) in enumerate(zip(reader.pages, texts))
                      if not text and len(page.images) > 0]
        if image_only:
            return {'ok': False, 'code': 'needs_ocr', 'pages': count,
                    'image_only_pages': image_only, 'partial_text': any(texts)}
        if not any(texts):
            return {'ok': False, 'code': 'empty_document', 'pages': count}
        return {'ok': True, 'code': 'accepted', 'pages': count,
                'text_characters': sum(map(len, texts))}
    except Exception as exc:
        return {'ok': False, 'code': 'invalid_pdf', 'error_type': type(exc).__name__}


def _raster_page(page, width: float, height: float) -> bytes:
    from reportlab.pdfgen import canvas
    from reportlab.lib.utils import ImageReader
    image = page.render(scale=1.25).to_pil()
    output = BytesIO()
    c = canvas.Canvas(output, pagesize=(width, height), invariant=1)
    c.drawImage(ImageReader(image), 0, 0, width=width, height=height)
    c.showPage(); c.save()
    return output.getvalue()


def build_fixtures(corpus: Corpus | str | Path | None = None) -> list[dict]:
    from pypdf import PdfReader, PdfWriter
    import pypdfium2 as pdfium
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import A4
    from .render import render_document

    corpus = corpus if isinstance(corpus, Corpus) else Corpus(corpus)
    out = corpus.root / 'fixtures'; out.mkdir(parents=True, exist_ok=True)
    rows = []

    def record(id: str, expected: str, description: str, **extra: Any) -> None:
        path = out / f'{id}.pdf'
        result = preflight(path)
        if result['code'] != expected:
            raise RuntimeError(f'{id}: expected {expected}, got {result}')
        rows.append({'id': id, 'path': f'fixtures/{id}.pdf', 'suite': 'X',
                     'sha256': sha256(path), 'expected_code': expected,
                     'description': description, 'observed_reference_result': result, **extra})

    w = PdfWriter()
    for _ in range(3): w.add_blank_page(width=A4[0], height=A4[1])
    with (out / 'X01.pdf').open('wb') as f: w.write(f)
    record('X01', 'empty_document', 'Valid PDF with three truly blank pages.')

    parent = corpus.get('S02').pdf_path
    with pdfium.PdfDocument(parent) as rendered:
        w = PdfWriter()
        for page in rendered:
            size = page.get_size()
            raw = _raster_page(page, *size)
            w.add_page(PdfReader(BytesIO(raw)).pages[0])
            page.close()
        with (out / 'X02.pdf').open('wb') as f: w.write(f)
    record('X02', 'needs_ocr', 'All six pages of S02 rasterized; no hidden text layer.', parent_id='S02')

    reader = PdfReader(parent); w = PdfWriter()
    with pdfium.PdfDocument(parent) as rendered:
        for i, page in enumerate(reader.pages):
            if i == 1:
                rpage = rendered[i]
                raw = _raster_page(rpage, *rpage.get_size()); rpage.close()
                w.add_page(PdfReader(BytesIO(raw)).pages[0])
            else: w.add_page(page)
    with (out / 'X03.pdf').open('wb') as f: w.write(f)
    record('X03', 'needs_ocr', 'Only physical page 2 of S02 is rasterized.', parent_id='S02')

    (out / 'X04.pdf').write_bytes(b'%PDF-1.7\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n% deliberately truncated fixture\n')
    record('X04', 'invalid_pdf', 'Tiny truncated file; not a resource-exhaustion payload.')

    w = PdfWriter(); w.append(str(corpus.get('S08').pdf_path))
    # Only a test of the encrypted-file branch, not a recommended cipher policy.
    w.encrypt('docqa-test-only', algorithm='RC4-128')
    with (out / 'X05.pdf').open('wb') as f: w.write(f)
    record('X05', 'password_required', 'Encrypted S08; password is intentionally public.',
           fixture_password='docqa-test-only', parent_id='S08')

    w = PdfWriter(); w.append(str(corpus.get('S12').pdf_path))
    w.add_blank_page(width=A4[0], height=A4[1])
    with (out / 'X06.pdf').open('wb') as f: w.write(f)
    record('X06', 'too_many_pages', 'S12 plus one page: exactly 101 physical pages.', parent_id='S12')

    source = {'id':'X07','title':'Краткое уведомление','family_id':'X07','split':'dev',
              'kind':'synthetic','facts':{'days':2}, 'pages':[
                  {'title':'Краткое уведомление','section':None,
                   'blocks':[{'id':'x07-rule','text':'Получение пакета подтверждается в течение двух рабочих дней с регистрации.'}]}]}
    render_document(source, out / 'X07.pdf', out / 'X07_source')
    record('X07', 'accepted', 'Tiny one-page document; one chunk only for the declared >=512-character budget.',
           chunking_assumption={'minimum_character_budget':512,'split_by_page':True},
           question='В какой срок подтверждается получение пакета?',
           reference_answer='В течение двух рабочих дней с регистрации.',
           source_dir='fixtures/X07_source')

    (out / 'X08.pdf').write_text('<!doctype html><title>Not a PDF</title><p>Wrong media type fixture.</p>', encoding='utf-8')
    record('X08', 'unsupported_media_type', 'HTML bytes with a .pdf filename.')
    write_json(out / 'manifest.json', {'version':'0.1.0','policy':'text_only_conservative_v1','fixtures':rows,
               'warning':'Do not upload this directory as a successful document-QA corpus. X04/X08 are intentionally invalid.'})
    return rows


def verify_fixtures(corpus: Corpus | str | Path | None = None) -> dict:
    corpus = corpus if isinstance(corpus, Corpus) else Corpus(corpus)
    rows = read_json(corpus.root / 'fixtures/manifest.json')['fixtures']; results = []
    for row in rows:
        path = corpus.root / row['path']; result = preflight(path)
        results.append({'id':row['id'],'ok':sha256(path)==row['sha256'] and result['code']==row['expected_code'],
                        'expected':row['expected_code'],'actual':result['code']})
    return {'ok':all(r['ok'] for r in results),'results':results}
