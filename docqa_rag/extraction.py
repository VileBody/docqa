"""Strict text-only extraction; immutable code-point coordinates, physical pages.

No OCR or normalization. Any raster content is rejected conservatively because
it may contain omitted evidence. Text in vector diagrams retains coordinates;
this profile does not interpret geometric relationships.
"""
from pathlib import Path
import pymupdf
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .types import Box, Page, Document, Chunk
from .util import digest, UnsupportedPDF

PARSER = 'pymupdf-'+pymupdf.VersionBind+'/rawdict-unsorted-v2-blank-pages'
TEXT_PARSER = 'utf8-exact-formfeed-pages-v1'

def extract_text(path, doc_id, expected_sha256=None):
    """Exact UTF-8, no newline/Unicode normalization; FF separates logical pages.

    Legacy internal pdf_sha256 holds source-file SHA for both formats. Service
    responses expose it as source_sha256 plus explicit source_format.
    """
    data=Path(path).read_bytes(); sha=digest(data)
    if len(data)>30*1024*1024 or (expected_sha256 and sha!=expected_sha256):
        raise ValueError('Text size/hash mismatch')
    text=data.decode('utf-8', errors='strict')
    if not text.strip() or '\x00' in text or '\ufffd' in text or text.startswith('\ufeff'):
        raise ValueError('Empty, BOM or damaged UTF-8 text')
    if any(ord(c)<32 and c not in '\n\r\t\f' for c in text):
        raise ValueError('Binary/control character in text')
    parts=text.split('\f')
    if len(parts)>100:raise ValueError('Text exceeds 100 logical pages')
    pages=[];offset=0
    for i,part in enumerate(parts,1):
        pages.append(Page(page=i,start=offset,end=offset+len(part),text=part,boxes=[],
                          extraction_status='text' if part.strip() else 'blank'))
        offset+=len(part)+1
    return Document(doc_id=doc_id,doc_title=doc_id,pdf_sha256=sha,text_sha256=digest(data),
                    text=text,pages=pages,parser=TEXT_PARSER)

def extract_source(path, doc_id, expected_sha256=None):
    if Path(path).suffix.lower()=='.txt':return extract_text(path,doc_id,expected_sha256)
    if Path(path).suffix.lower()=='.pdf':return extract_pdf(path,doc_id,expected_sha256)
    raise ValueError('Supported source formats: TXT, optional PDF')

def extract_pdf(path: str | Path, doc_id: str, expected_sha256: str | None = None) -> Document:
    data=Path(path).read_bytes()
    if len(data)>30*1024*1024 or not data.startswith(b'%PDF-'):
        raise UnsupportedPDF('Not a PDF or exceeds 30 MiB')
    sha=digest(data)
    if expected_sha256 and sha != expected_sha256: raise UnsupportedPDF('PDF hash mismatch')
    pages=[]; pieces=[]; offset=0
    try:
        with pymupdf.open(stream=data, filetype='pdf') as pdf:
            if pdf.needs_pass or pdf.is_repaired or not 1<=len(pdf)<=100:
                raise UnsupportedPDF('Encrypted, repaired or outside 1..100 pages')
            for number, page in enumerate(pdf,1):
                if page.get_image_info():
                    raise UnsupportedPDF(f'Page {number}: raster content requires visual/OCR profile')
                text=''; boxes=[]
                for block in page.get_text('rawdict', sort=False)['blocks']:
                    for line in block.get('lines',[]):
                        for span in line['spans']:
                            for char in span['chars']:
                                value=char['c']; start=offset+len(text); text+=value
                                boxes.append(Box(start=start,end=offset+len(text),bbox=char['bbox']))
                        text+='\n'
                blank=not text.strip()
                if blank and (page.get_drawings() or list(page.annots() or []) or list(page.widgets() or [])):
                    raise UnsupportedPDF(f'Page {number}: non-text content on empty text page')
                if '\ufffd' in text or '\x00' in text:
                    raise UnsupportedPDF(f'Page {number}: empty or damaged text layer')
                if any(ord(c)<32 and c not in '\n\t\r' for c in text):
                    raise UnsupportedPDF(f'Page {number}: invalid control characters')
                pages.append(Page(page=number,start=offset,end=offset+len(text),text=text,boxes=boxes,extraction_status='blank' if blank else 'text'))
                pieces.append(text); offset+=len(text)
                if number < len(pdf): pieces.append('\f'); offset+=1
    except UnsupportedPDF: raise
    except Exception as e: raise UnsupportedPDF('PDF extraction failed: '+type(e).__name__) from e
    text=''.join(pieces)
    if not text.strip():raise UnsupportedPDF('Entire document has no usable text')
    return Document(doc_id=doc_id,doc_title=doc_id,pdf_sha256=sha,text_sha256=digest(text.encode()),text=text,pages=pages,parser=PARSER)

def split_document(doc: Document, size: int = 1200, overlap: int = 160) -> list[Chunk]:
    splitter=RecursiveCharacterTextSplitter(chunk_size=size,chunk_overlap=overlap,
        separators=['\n\n','\n',' ',''],keep_separator=True,strip_whitespace=False,add_start_index=True)
    result=[]
    for page in doc.pages:
        for part in splitter.create_documents([page.text]):
            a=page.start+part.metadata['start_index']; b=a+len(part.page_content)
            if a<page.start or doc.text[a:b]!=part.page_content: raise ValueError('Splitter offset mismatch')
            if not part.page_content.strip(): continue
            result.append(Chunk(source_id=digest([doc.pdf_sha256,doc.text_sha256,a,b]),doc_id=doc.doc_id,
                doc_title=doc.doc_title,chunk_index=len(result),page_from=page.page,page_to=page.page,
                text=doc.text[a:b],char_start=a,char_end=b,text_sha256=doc.text_sha256,pdf_sha256=doc.pdf_sha256))
    return result
