"""Small read-only search tool. Application owns scope, snapshot and hard ceilings.

Returned source text is evidence, never an inner generator's answer. Controllers
may cite an already returned receipt to ground a subsequent query's new anchor.
"""
import re
import time
from dataclasses import dataclass
from pydantic import Field
from .types import Contract, Claim, Reference
from .flow import plan_query
from .planning import date_anchors
from .evidence import resolve_claim
from .selection import select_pack
from .tokenization import count
from .util import digest, BudgetExceeded, ModelUnavailable

TOOL_VERSION='search-evidence-v1'

class SearchRequest(Contract):
    query:str=Field(min_length=1,max_length=2048)
    source_evidence_ids:list[str]=Field(default_factory=list,max_length=32)

class InvalidToolRequest(ValueError):pass

@dataclass(frozen=True)
class SearchLimits:
    searches:int=2
    views:int=2
    rerank_pairs:int=48
    source_fragments:int=12
    returned_tokens:int=8192
    wall_seconds:float=120

    def __post_init__(self):
        if any(v<=0 for v in vars(self).values()):raise ValueError('Positive tool limits required')

def anchor_provenance(question,query,receipts):
    """Only explicit anchors/dates, not a proof of general query meaning."""
    contexts=[('original_question',question,None)]+[(key,r['fragment']['text'],r) for key,r in receipts.items()]
    used=[]
    normalized_query=query
    for start,end,date in reversed(date_anchors(query)):
        match=next(((key,text,r,a,b) for key,text,r in contexts for a,b,d in date_anchors(text) if d==date),None)
        if match is None:raise InvalidToolRequest('Unobserved date anchor')
        key,text,r,a,b=match
        used.append(_provenance(date,key,r,a,b,text[a:b]))
        normalized_query=normalized_query[:start]+' DATE '+normalized_query[end:]
    for anchor in plan_query(normalized_query).anchors:
        match=next(((key,text,r) for key,text,r in contexts if anchor in plan_query(text).anchors),None)
        if match is None:raise InvalidToolRequest('Unobserved numeric/identifier anchor')
        key,text,r=match;a=text.index(anchor)
        used.append(_provenance(anchor,key,r,a,a+len(anchor),anchor))
    return used

def _provenance(anchor,key,receipt,a,b,quote):
    value={'anchor':anchor,'origin':key,'quote':quote,'start':a,'end':b}
    if receipt:
        fragment=receipt['fragment']
        value.update(source_id=fragment['source_id'],text_sha256=fragment['text_sha256'],
                     char_start=fragment['char_start']+a,char_end=fragment['char_start']+b,
                     snapshot=receipt['snapshot'])
    return value

class EvidenceSearch:
    def __init__(self,index,embedder,reranker,profile,*,doc_id,question,limits=SearchLimits()):
        if doc_id not in index.documents:raise InvalidToolRequest('Application document scope missing')
        plan_query(question)
        self.index=index;self.embedder=embedder;self.reranker=reranker;self.profile=profile.model_copy(deep=True)
        self.doc_id=doc_id;self.question=question;self.snapshot=index.fingerprint;self.limits=limits
        self.started=time.monotonic();self.receipts={};self.history=[]
        self.actual={'searches':0,'views':0,'rerank_pairs':0,'source_fragments':0,'returned_tokens':0}
        self.stopped=None

    def _time(self):
        if time.monotonic()-self.started>=self.limits.wall_seconds:raise TimeoutError('Tool deadline')

    def search(self,request):
        if self.stopped:raise InvalidToolRequest('Session stopped: '+self.stopped)
        try:
            req=SearchRequest.model_validate(request)
            if not req.query.strip():raise InvalidToolRequest('Empty query')
            if len(set(req.source_evidence_ids))!=len(req.source_evidence_ids):raise InvalidToolRequest('Duplicate receipt')
            if any(key not in self.receipts for key in req.source_evidence_ids):raise InvalidToolRequest('Unknown or not-yet-observed receipt')
            selected={key:self.receipts[key] for key in req.source_evidence_ids}
            provenance=anchor_provenance(self.question,req.query,selected)
        except ValueError as exc:
            self.stopped='invalid_tool_request'
            self.history.append({'status':self.stopped,'error_type':type(exc).__name__,'actual':dict(self.actual),'evidence':[]})
            raise InvalidToolRequest('Invalid tool request: '+str(exc)) from exc
        event={'request':req.model_dump(),'anchor_provenance':provenance,'snapshot':self.snapshot,'tool_version':TOOL_VERSION}
        start=time.monotonic()
        try:
            self._time()
            if self.index.fingerprint!=self.snapshot:raise InvalidToolRequest('Snapshot changed')
            if self.actual['searches']>=self.limits.searches or self.actual['views']>=self.limits.views:
                raise BudgetExceeded('Search/view ceiling')
            # Reserve worst-case rerank work before any model-backed retrieval.
            if self.actual['rerank_pairs']+self.profile.candidate_limit>self.limits.rerank_pairs:
                raise BudgetExceeded('Rerank-pair ceiling')
            self.actual['searches']+=1;self.actual['views']+=1
            candidates=self.index.search(self.doc_id,plan_query(req.query),self.embedder,self.profile.candidate_limit,'hybrid',self.profile.fusion)
            self._time()
            if len(candidates)>self.profile.candidate_limit:raise InvalidToolRequest('Candidate ceiling violated')
            self.actual['rerank_pairs']+=len(candidates)
            ranked=self.reranker.rank(req.query,candidates)
            source={h.chunk.source_id:h.chunk.model_dump() for h in candidates}
            if len(ranked)!=len(candidates) or len({h.chunk.source_id for h in ranked})!=len(ranked) or any(source.get(h.chunk.source_id)!=h.chunk.model_dump() for h in ranked):
                raise InvalidToolRequest('Reranker altered or lost source ranges')
            pack,selection=select_pack(candidates,ranked,self.profile)
            self._time()
            if self.index.fingerprint!=self.snapshot:raise InvalidToolRequest('Snapshot changed')
            returned=[]
            for hit in pack:
                chunk=hit.chunk
                # Identical document/scope/hash/page/range verification as Flow citations.
                resolve_claim(Claim(text='source validation',references=[Reference(source_id=chunk.source_id,start=0,end=len(chunk.text))]),pack,self.index.documents,self.doc_id)
                key=digest({'snapshot':self.snapshot,'fragment':chunk.model_dump()})
                if key in self.receipts:continue
                tokens=count(chunk.text)
                if self.actual['source_fragments']+1>self.limits.source_fragments or self.actual['returned_tokens']+tokens>self.limits.returned_tokens:
                    self.stopped='budget_exhausted';break
                receipt={'evidence_id':key,'snapshot':self.snapshot,'fragment':chunk.model_dump(),'rerank_score':hit.rerank_score}
                self.receipts[key]=receipt;returned.append(receipt)
                self.actual['source_fragments']+=1;self.actual['returned_tokens']+=tokens
            if not returned and not self.stopped:self.stopped='repeated_evidence' if pack else 'no_new_evidence'
            event.update(status=self.stopped or 'success',evidence=returned,selection=selection)
        except BudgetExceeded:
            self.stopped='budget_exhausted';event.update(status=self.stopped,evidence=[])
        except (TimeoutError,ModelUnavailable) as exc:
            self.stopped='timeout' if isinstance(exc,TimeoutError) else 'model_unavailable'
            event.update(status=self.stopped,evidence=[],error_type=type(exc).__name__)
        except Exception as exc:
            self.stopped='invalid_tool_result';event.update(status=self.stopped,evidence=[],error_type=type(exc).__name__)
            raise
        finally:
            event.update(actual=dict(self.actual),wall_s=time.monotonic()-start)
            self.history.append(event)
        return event
