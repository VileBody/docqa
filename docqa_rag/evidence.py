"""Issued line spans and strict reversible quote/range resolution."""
import re
from .types import Citation
from .util import InvalidEvidence

CITATION_RANGE_VERSION='contiguous-line-range-v2'

class EvidenceIssue(InvalidEvidence):
    def __init__(self,code,message):super().__init__(message);self.code=code

def spans(chunk,include_blank=False):
    result={};start=0
    for line in chunk.text.splitlines(keepends=True):
        end=start+len(line)
        if line.strip() or include_blank:result['L'+str(len(result)+1)]={'start':start,'end':end,'text':line}
        start=end
    return result

def issued_sources(hits,mode='quote'):
    sources=[];catalog={}
    for i,h in enumerate(hits,1):
        alias='S'+str(i);c=h.chunk;lines=spans(c,include_blank=mode=='span_range')
        catalog[alias]={'source_id':c.source_id,'page':c.page_from,'char_start':c.char_start,'char_end':c.char_end,'spans':lines}
        if mode in {'span','span_range'}:
            entries=[{'span_id':k,'text':v['text']} for k,v in lines.items()]
            if mode=='span_range':
                entries=[{'span_id':k,**v} for k,v in lines.items()]
                # Every line, including blanks, appears exactly once; no duplicated
                # full-text payload and no untransmitted characters inside a range.
                sources.append({'source_id':alias,'spans':entries,'range_contract':CITATION_RANGE_VERSION,'boundary_completeness':'not_guaranteed; source_is_a_fragment'})
            else:sources.append({'source_id':alias,'spans':entries})
        else:sources.append({'source_id':alias,'text':c.text})
    return sources,catalog

def expand_span_ranges(wire_draft, catalog):
    """Convert issued start/end IDs to ONE real contiguous range, never joined text."""
    data=wire_draft.model_dump()
    for claim in data['claims']:
        resolved=[]
        for ref in claim['references']:
            source=catalog.get(ref['source_id'])
            if source is None:raise EvidenceIssue('unknown_source_id','Unissued source alias')
            start=source['spans'].get(ref['start_span_id']);end=source['spans'].get(ref['end_span_id'])
            if start is None or end is None:raise EvidenceIssue('unknown_span','Unissued range endpoint')
            if start['start']>end['start']:raise EvidenceIssue('invalid_range','Reversed span range')
            resolved.append({'source_id':ref['source_id'],'start':start['start'],'end':end['end']})
        claim['references']=resolved
    return data

def resolve_claim(claim,hits,documents,doc_id):
    sources={h.chunk.source_id:h.chunk for h in hits};citations=[]
    if doc_id not in documents:raise EvidenceIssue('unknown_document','Unknown document')
    doc=documents[doc_id]
    for ref in claim.references:
        c=sources.get(ref.source_id)
        if c is None:raise EvidenceIssue('source_not_in_pack','Source ID not in transmitted pack')
        if c.doc_id!=doc_id or c.text_sha256!=doc.text_sha256 or c.pdf_sha256!=doc.pdf_sha256:
            raise EvidenceIssue('wrong_source','Source does not belong to selected immutable document')
        if not 0<=c.char_start<c.char_end<=len(doc.text) or doc.text[c.char_start:c.char_end]!=c.text:
            raise EvidenceIssue('invalid_source_range','Source range altered')
        if ref.page is not None and ref.page!=c.page_from:raise EvidenceIssue('page_mismatch','Requested page disagrees with source')
        if ref.span_id is not None:
            span=spans(c).get(ref.span_id)
            if span is None:raise EvidenceIssue('unknown_span','Unissued span ID')
            if ref.start is not None or ref.end is not None or ref.occurrence is not None:raise EvidenceIssue('conflicting_selector','Mixed span and range/occurrence')
            a,b=span['start'],span['end'];policy='issued_span'
            if ref.quote is not None and ref.quote!=c.text[a:b]:raise EvidenceIssue('quote_mismatch','Quote differs from issued span')
        elif ref.start is not None or ref.end is not None:
            a,b=ref.start,ref.end;policy='explicit_range'
            if a is None or b is None or not 0<=a<b<=len(c.text):raise EvidenceIssue('invalid_range','Range outside transmitted source')
            if ref.occurrence is not None:raise EvidenceIssue('conflicting_selector','Range and occurrence both supplied')
            if ref.quote is not None and ref.quote!=c.text[a:b]:raise EvidenceIssue('quote_mismatch','Quote differs from explicit range')
        else:
            if not ref.quote or not ref.quote.strip():raise EvidenceIssue('missing_quote','Missing quote/span/range')
            matches=list(re.finditer('(?='+re.escape(ref.quote)+')',c.text));bounds=[(m.start(),m.start()+len(ref.quote)) for m in matches];policy='exact'
            if not bounds:
                pattern=r'\s+'.join(re.escape(t) for t in re.split(r'\s+',ref.quote))
                bounds=[m.span() for m in re.finditer(pattern,c.text)];policy='whitespace_runs'
            if not bounds:raise EvidenceIssue('quote_not_found','Quote not present in transmitted source')
            if ref.occurrence is None:
                if len(bounds)!=1:raise EvidenceIssue('ambiguous_quote','Multiple quote occurrences; explicit selector required')
                a,b=bounds[0]
            else:
                if ref.occurrence>len(bounds):raise EvidenceIssue('invalid_occurrence','Occurrence not present')
                a,b=bounds[ref.occurrence-1]
        start,end=c.char_start+a,c.char_start+b
        page=next((p for p in doc.pages if p.start<=start<end<=p.end),None)
        if page is None or page.page!=c.page_from or c.page_from!=c.page_to:raise EvidenceIssue('page_range_mismatch','Page and range inconsistent')
        citations.append(Citation(match_policy=policy,source_id=c.source_id,document_id=doc_id,page=page.page,text=doc.text[start:end],
            char_start=start,char_end=end,text_sha256=doc.text_sha256,pdf_sha256=doc.pdf_sha256,boxes=[b for b in page.boxes if b.start<end and b.end>start]))
    return citations
