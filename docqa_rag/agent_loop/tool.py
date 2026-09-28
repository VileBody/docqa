"""Hybrid Qdrant search with separate candidates, observations and final evidence."""
import time
from ..types import Claim, Reference, Hit
from ..flow import plan_query
from ..selection import select_pack
from ..evidence import resolve_claim
from ..search_evidence import SearchRequest, anchor_provenance, InvalidToolRequest
from ..util import digest
from .contracts import Limits, ScopeViolation

class SearchCap(RuntimeError): pass

class SearchTool:
    def __init__(self,index,embedder,reranker,profile,doc_id,question,limits=None):
        self.limits=limits or Limits();self.index=index;self.embedder=embedder;self.reranker=reranker
        self.profile=profile.model_copy(update={'candidate_limit':self.limits.candidates,'rerank_limit':self.limits.visible})
        if doc_id not in index.documents:raise ScopeViolation('Document not in application scope')
        self.doc_id=doc_id;self.question=question;self.snapshot=index.fingerprint
        self.visible={};self.reservoir={};self.cache={};self.history=[];self.searches=0;self.logical_pairs=0
        self.started=time.monotonic();self.no_progress=0
    def check(self):
        if self.index.fingerprint!=self.snapshot:raise ScopeViolation('Snapshot changed')
        if time.monotonic()-self.started>self.limits.wall_seconds:raise TimeoutError('Episode deadline')
    def validate(self,request):
        req=SearchRequest.model_validate(request)
        if not req.query.strip():raise ValueError('Empty query')
        if len(set(req.source_evidence_ids))!=len(req.source_evidence_ids) or any(k not in self.visible for k in req.source_evidence_ids):
            raise ScopeViolation('Unknown or hidden source reference')
        try: anchors=anchor_provenance(self.question,req.query,{k:self.visible[k] for k in req.source_evidence_ids})
        except InvalidToolRequest as exc:raise ScopeViolation(str(exc)) from exc
        return req,anchors
    def identity(self,req):
        return {'query':req.query,'views':[req.query],'scope':self.doc_id,'snapshot':self.snapshot,
                'filters':{'doc_id':self.doc_id,'ready':True,'snapshot':self.snapshot},
                'profile':self.profile.model_dump(),'version':'search-v2'}
    def _authentic(self,hits):
        for h in hits:
            resolve_claim(Claim(text='validate source',references=[Reference(source_id=h.chunk.source_id,start=0,end=len(h.chunk.text))]),hits,self.index.documents,self.doc_id)
    def _rank(self,query,candidates):
        self.logical_pairs+=len(candidates)
        if self.logical_pairs>self.limits.logical_pairs:raise SearchCap('Logical pair cap')
        ranked=self.reranker.rank(query,candidates)
        expected={h.chunk.source_id:h.chunk.model_dump() for h in candidates}
        if len(ranked)!=len(candidates) or len({h.chunk.source_id for h in ranked})!=len(ranked) or any(expected.get(h.chunk.source_id)!=h.chunk.model_dump() for h in ranked):
            raise ScopeViolation('Reranker changed source identity')
        return ranked
    def search(self,request):
        self.check();req,anchors=self.validate(request);key=digest(self.identity(req))
        if key in self.cache:
            event={**self.cache[key],'outcome':'duplicate_request','cached_outcome':self.cache[key]['outcome'],'cache_hit':True,'new_evidence_ids':[]}
            self.history.append(event);return event
        if self.searches>=self.limits.searches:raise SearchCap('Search budget reached')
        self.searches+=1
        candidates=self.index.search(self.doc_id,plan_query(req.query),self.embedder,self.limits.candidates,'hybrid',self.profile.fusion)
        if len(candidates)>self.limits.candidates:raise ScopeViolation('Candidate cap')
        self._authentic(candidates)
        ranked=self._rank(req.query,candidates);self.check()
        byid={h.chunk.source_id:(i+1,h.rerank_score) for i,h in enumerate(ranked)}
        for i,h in enumerate(candidates):
            c=h.chunk;rkey=digest({'doc_id':c.doc_id,'hash':c.text_sha256,'start':c.char_start,'end':c.char_end,'snapshot':self.snapshot})
            if rkey not in self.reservoir and len(self.reservoir)<self.limits.reservoir:
                self.reservoir[rkey]={'hit':h.model_dump(),'origins':[]}
            if rkey in self.reservoir:
                self.reservoir[rkey]['origins'].append({'query_sha256':key,'retrieval_rank':i+1,'retrieval_score':h.retrieval_score,'rerank_rank':byid[c.source_id][0],'rerank_score':byid[c.source_id][1]})
        pack,selection=select_pack(candidates,ranked,self.profile)
        evidence=[];new=[]
        for h in pack:
            eid=digest({'snapshot':self.snapshot,'fragment':h.chunk.model_dump()})
            receipt={'evidence_id':eid,'snapshot':self.snapshot,'fragment':h.chunk.model_dump()}
            if eid not in self.visible:new.append(eid)
            self.visible[eid]=receipt;evidence.append(receipt)
        outcome='empty_candidates' if not candidates else 'empty_selected' if not pack else 'no_new_visible_sources' if not new else 'ok'
        self.no_progress=0 if new else self.no_progress+1
        event={'request':req.model_dump(),'outcome':outcome,'evidence':evidence,'new_evidence_ids':new,'cache_hit':False,
               'anchor_provenance':anchors,'selection':selection,'searches':self.searches,'no_progress_count':self.no_progress}
        self.cache[key]=event;self.history.append(event);return event
    def final_pack(self):
        self.check()
        candidates=[Hit.model_validate(r['hit']) for r in self.reservoir.values()]
        self._authentic(candidates)
        ranked=self._rank(self.question,candidates)
        pack,selection=select_pack(candidates,ranked,self.profile)
        self.check();return pack,selection
    def trace(self):
        return {'searches':self.history,'visible_evidence':self.visible,'candidate_reservoir':self.reservoir,
                'reservoir_overflow_policy':'first_seen_stable_scope_range; B*C capacity',
                'searches_executed':self.searches,'logical_rerank_pairs':self.logical_pairs,'snapshot':self.snapshot}
