"""Explicit official-source acquisition and gated evidence review.

Network is never used by Corpus(). No synthetic stand-ins for official PDFs.
A successful download is not synonymous with validated annotations.
"""
from __future__ import annotations
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, build_opener, HTTPRedirectHandler
from urllib.error import HTTPError, URLError
import os
import tempfile
import time
from typing import Any
from .core import Corpus
from .build import register
from .util import read_json, read_jsonl, write_json, write_jsonl, write_text, sha256, text_hash, PAGE_SEPARATOR

ALLOWED_HOSTS={'www.nalog.gov.ru','nalog.gov.ru','data.nalog.ru','www.cbr.ru','cbr.ru'}

def _checked_url(url: str) -> str:
    parsed=urlparse(url)
    if parsed.scheme!='https' or parsed.hostname not in ALLOWED_HOSTS or parsed.username or parsed.password or parsed.port not in (None,443):
        raise ValueError('Only the allowlisted official HTTPS origins on port 443 are supported')
    return url

class _OfficialRedirects(HTTPRedirectHandler):
    def redirect_request(self,req,fp,code,msg,headers,newurl):
        return super().redirect_request(req,fp,code,msg,headers,_checked_url(newurl))


def _freeze(corpus: Corpus, document_id: str, temporary: Path, provenance: dict[str,Any]) -> dict:
    from pypdf import PdfReader
    d=corpus.get(document_id)
    with temporary.open('rb') as f:
        if not f.read(8).startswith(b'%PDF-'):raise ValueError('Downloaded bytes are not a PDF')
    reader=PdfReader(temporary,strict=False)
    if reader.is_encrypted:raise ValueError('Unexpected encrypted official document')
    if len(reader.pages)!=d.metadata['pages']:
        raise ValueError(f'Source may have changed: expected {d.metadata["pages"]} pages, got {len(reader.pages)}')
    pages=[];parts=[];offset=0
    for i,page in enumerate(reader.pages,1):
        # Frozen extraction, no reflow/normalization silently changing offsets.
        text=page.extract_text() or ''
        pages.append({'page':i,'printed_page':None,'char_start':offset,'char_end':offset+len(text),'text':text})
        parts.append(text);offset+=len(text)+len(PAGE_SEPARATOR)
    canonical=PAGE_SEPARATOR.join(parts);digest=sha256(temporary)
    source=d.source_dir;source.mkdir(parents=True,exist_ok=True)
    d.pdf_path.parent.mkdir(parents=True,exist_ok=True)
    os.replace(temporary,d.pdf_path)
    write_text(source/'canonical.txt',canonical);write_json(source/'pages.json',pages)
    write_json(source/'provenance.json',{
        **provenance,'document_id':document_id,'sha256':digest,'canonical_sha256':text_hash(canonical),
        'extraction':'pypdf default text extraction; frozen verbatim per page',
        'page_separator':PAGE_SEPARATOR,'physical_pages':len(pages),
        'independent_human_review':False,'retrieved_at':datetime.now(timezone.utc).isoformat(),
        'known_hash_before_first_download':False})
    item={**d.metadata,'status':'downloaded_needs_review','original_in_archive':True,
          'sha256':digest,'canonical_sha256':text_hash(canonical)}
    register(corpus.root,item)
    return {'id':document_id,'ok':True,'status':item['status'],'sha256':digest,'pages':len(pages)}


def fetch_public(corpus: Corpus | str | Path | None = None, *, ids: list[str] | None = None,
                 timeout: float = 30, attempts: int = 3, max_bytes: int = 30_000_000) -> dict:
    """Download missing originals only. Never overwrite an established snapshot.

    Bounded reads/retries; HTTPS redirects remain on known official hosts. Network
    failures are returned per document, with a false aggregate ok flag.
    """
    corpus=corpus if isinstance(corpus,Corpus) else Corpus(corpus)
    if not 1<=attempts<=10 or timeout<=0 or max_bytes<1:raise ValueError('Invalid download limits')
    selected=list(corpus.documents(suite='R',available_only=False))
    if ids:
        unknown=set(ids)-{d.id for d in selected}
        if unknown:raise ValueError(f'Unknown public IDs: {sorted(unknown)}')
        selected=[d for d in selected if d.id in ids]
    opener=build_opener(_OfficialRedirects());results=[]
    for d in selected:
        if d.available:
            expected=d.metadata.get('sha256')
            if not expected:
                results.append({'id':d.id,'ok':False,'error':'Unregistered local file. Use import-public --file; do not infer provenance.'})
            elif sha256(d.pdf_path)!=expected:
                results.append({'id':d.id,'ok':False,'error':'Existing snapshot hash mismatch; no replacement attempted.'})
            else:results.append({'id':d.id,'ok':True,'status':'already_present','sha256':expected})
            continue
        url=_checked_url(d.metadata['source_url']);last=''
        for attempt in range(attempts):
            tmp=None
            try:
                d.pdf_path.parent.mkdir(parents=True,exist_ok=True)
                fd,name=tempfile.mkstemp(prefix='.download-',suffix='.pdf',dir=d.pdf_path.parent);os.close(fd);tmp=Path(name)
                request=Request(url,headers={'User-Agent':'DocQACorpusKit/0.2 (+offline evaluation corpus)','Accept':'application/pdf'})
                with opener.open(request,timeout=timeout) as response,tmp.open('wb') as out:
                    final_url=_checked_url(response.geturl());total=0
                    content_length=response.headers.get('Content-Length')
                    if content_length and int(content_length)>max_bytes:raise ValueError('Source exceeds byte limit')
                    while block:=response.read(64*1024):
                        total+=len(block)
                        if total>max_bytes:raise ValueError('Source exceeds byte limit')
                        out.write(block)
                    headers={'etag':response.headers.get('ETag'),'last_modified':response.headers.get('Last-Modified')}
                results.append(_freeze(corpus,d.id,tmp,{'source_url':url,'resolved_url':final_url,'method':'official_https',**headers}))
                break
            except (ValueError,HTTPError,URLError,OSError) as exc:
                last=f'{type(exc).__name__}: {exc}'
                transient=isinstance(exc,(URLError,OSError)) and not isinstance(exc,HTTPError)
                if isinstance(exc,HTTPError):transient=exc.code in (408,429,500,502,503,504)
                if not transient or attempt+1==attempts:
                    results.append({'id':d.id,'ok':False,'status':'not_downloaded','error':last});break
                time.sleep(min(2**attempt,4))
            except Exception as exc:
                results.append({'id':d.id,'ok':False,'status':'invalid_source','error':f'{type(exc).__name__}: {exc}'});break
            finally:
                if tmp is not None:tmp.unlink(missing_ok=True)
    return {'ok':all(r['ok'] for r in results),'results':results}


def import_public_file(corpus: Corpus | str | Path, document_id: str, path: str | Path) -> dict:
    """Register an original manually downloaded from the declared official URL.

    Origin is user-attested in this mode; a matching page count is not provenance.
    """
    import shutil
    corpus=corpus if isinstance(corpus,Corpus) else Corpus(corpus);d=corpus.get(document_id)
    if d.metadata['suite']!='R':raise ValueError('Only R documents can be imported')
    if d.available and d.metadata.get('sha256'):raise FileExistsError('Existing registered snapshot is immutable')
    p=Path(path).resolve()
    if not p.is_file() or p.stat().st_size>30_000_000:raise ValueError('Missing or oversized original')
    d.pdf_path.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(prefix='.import-',suffix='.pdf',dir=d.pdf_path.parent);os.close(fd);tmp=Path(name)
    try:
        shutil.copyfile(p,tmp)
        return _freeze(corpus,document_id,tmp,{'source_url':d.metadata['source_url'],'method':'user_import_origin_not_independently_verified'})
    finally:tmp.unlink(missing_ok=True)


def review_template(corpus: Corpus | str | Path, document_id: str, output: str | Path) -> dict:
    corpus=corpus if isinstance(corpus,Corpus) else Corpus(corpus);d=corpus.get(document_id)
    if d.metadata['suite']!='R' or not d.available:raise ValueError('Download/import this public document first')
    rows=[]
    for q in d.questions(ready_only=False):
        draft=q.get('draft_reference',{})
        rows.append({'id':q['id'],'question':q['question'],'exclude_reason':None,
                     'found':draft.get('found'),'answer':draft.get('answer',''),
                     'required_facts':[],'whole_document_reviewed':False,
                     'evidence':[{'page':p,'text':'','section':None} for p in draft.get('physical_pages',[])]})
    template={'document_id':d.id,'pdf_sha256':sha256(d.pdf_path),'reviewer':'',
              'reviewer_kind':None,'independent_of_author':False,
              'attests_supported_answers_and_complete_negative_review':False,'questions':rows,
              'instructions':'Fill exact quotes from canonical.txt; verify physical PDF pages and meaning. Use exclude_reason for image-only/unscorable cases; never invent offsets.'}
    write_json(Path(output),template);return template


def apply_review(corpus: Corpus | str | Path, review_path: str | Path) -> dict:
    corpus=corpus if isinstance(corpus,Corpus) else Corpus(corpus);review=read_json(Path(review_path));d=corpus.get(review['document_id'])
    if d.metadata['suite']!='R':raise ValueError('Public review applies to R only')
    if not review.get('reviewer','').strip() or review.get('attests_supported_answers_and_complete_negative_review') is not True:
        raise ValueError('Named reviewer and explicit attestation are required')
    kind=review.get('reviewer_kind')
    independent=review.get('independent_of_author')
    if kind not in ('human','agent') or type(independent) is not bool:
        raise ValueError('Explicit reviewer_kind human/agent and boolean independent_of_author required')
    if kind=='agent' and independent is True:
        raise ValueError('Agent review cannot attest independent HUMAN review; set independent_of_author=False')
    human_independent=(kind=='human' and independent is True)
    if sha256(d.pdf_path)!=review.get('pdf_sha256'):raise ValueError('Review belongs to another PDF snapshot')
    pages={p['page']:p for p in read_json(d.source_dir/'pages.json')};canonical=d.canonical_text;h=text_hash(canonical)
    original={q['id']:q for q in d.questions(ready_only=False)};incoming=review.get('questions',[])
    if {q['id'] for q in incoming}!=set(original) or len(incoming)!=len(original):raise ValueError('Review must cover each question exactly once')
    out=[];excluded=0
    for r in incoming:
        q=deepcopy(original[r['id']])
        if r.get('exclude_reason'):
            q.update(status='excluded_after_review',exclusion_reason=r['exclude_reason'],gold=None);out.append(q);excluded+=1;continue
        if type(r.get('found')) is not bool or not isinstance(r.get('answer'),str) or not r['answer'].strip():raise ValueError(f'{r["id"]}: missing answer/found')
        if not r['found'] and r.get('whole_document_reviewed') is not True:raise ValueError(f'{r["id"]}: negative requires whole-file review')
        evidence=[]
        for i,e in enumerate(r.get('evidence',[]),1):
            p=pages.get(e.get('page')) if type(e.get('page')) is int else None;quote=e.get('text','')
            if not p or not quote or p['text'].count(quote)!=1:raise ValueError(f'{r["id"]}: quote must occur exactly once on its stated page')
            start=p['char_start']+p['text'].index(quote)
            evidence.append({'id':f'{r["id"]}::e{i}','document_id':d.id,'text':quote,'page':e['page'],
                'printed_page':None,'section':e.get('section'),'char_start':start,'char_end':start+len(quote),
                'bbox':None,'canonical_sha256':h})
        if bool(evidence)!=r['found']:raise ValueError('Found answers need evidence; abstentions must not contain citations')
        q.update(status='ready_human_reviewed' if human_independent else 'ready_agent_or_author_reviewed',coordinate_system='frozen_pypdf_text_v1',
                 review={'reviewer':review['reviewer'],'reviewer_kind':kind,
                         'independent_human_review':human_independent,'whole_document_reviewed':bool(r.get('whole_document_reviewed'))},
                 gold={'found':r['found'],'mode':'answer' if r['found'] else 'abstain','answer':r['answer'],
                       'required_facts':r.get('required_facts',[]),'evidence':evidence,
                       'evidence_sets':[[e['id'] for e in evidence]] if evidence else [],'forbidden_answer_substrings':[]})
        out.append(q)
    write_jsonl(corpus.root/d.metadata['annotations'],out)
    write_json(d.source_dir/'review_record.json',review)
    register(corpus.root,{**d.metadata,'status':'reviewed','ready_questions':len(out)-excluded})
    return {'id':d.id,'ready_questions':len(out)-excluded,'excluded':excluded}
