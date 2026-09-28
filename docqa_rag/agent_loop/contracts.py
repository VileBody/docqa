"""Application-owned ceilings and public, mutable research state."""
from dataclasses import dataclass, asdict
from typing import Literal
from pydantic import Field, model_validator
from ..types import Contract
from ..search_evidence import SearchRequest

class ScopeViolation(ValueError):
    """Unsafe scope/provenance request; never offered a retry."""

@dataclass(frozen=True)
class Limits:
    searches: int = 4
    candidates: int = 24
    visible: int = 6
    no_progress: int = 2
    repairs: int = 1
    output_tokens: int = 1536
    final_output_tokens: int = 2048
    wall_seconds: int = 900
    cumulative_input_tokens: int = 160000
    def __post_init__(self):
        if self.searches not in (1,2,4,8) or not 1 <= self.candidates <= 200:
            raise ValueError('Unsupported search/candidate ceiling')
        if self.repairs != 1 or min(self.visible,self.no_progress,self.output_tokens,self.wall_seconds,self.cumulative_input_tokens)<=0:
            raise ValueError('Invalid episode ceilings')
    @property
    def reservoir(self): return self.searches * self.candidates
    @property
    def controller_calls(self): return self.searches + 1 + self.repairs
    @property
    def logical_pairs(self): return self.searches*self.candidates + self.reservoir
    @property
    def wire_pairs(self): return 3*self.logical_pairs
    def manifest(self):
        return {**asdict(self),'reservoir':self.reservoir,'controller_calls':self.controller_calls,
                'logical_pairs':self.logical_pairs,'wire_pairs':self.wire_pairs,
                'controller_output_reserve':self.controller_calls*self.output_tokens,
                'final_output_reserve':self.final_output_tokens,'wire_batch_limit':24,
                'transport_attempts_per_request':1,'uncertain_request_policy':'reconcile_no_automatic_retry'}

class NeedPatch(Contract):
    id: str = Field(min_length=1,max_length=80)
    question: str = Field(min_length=1,max_length=1000)
    status: Literal['open','supported','uncertain','superseded']
    parent_id: str | None = None
    reason: str = Field(min_length=1,max_length=1000)
    evidence_ids: list[str] = Field(default_factory=list,max_length=16)

class ConflictPatch(Contract):
    id: str = Field(min_length=1,max_length=80)
    description: str = Field(min_length=1,max_length=1000)
    status: Literal['open','resolved','superseded']
    evidence_ids: list[str] = Field(min_length=2,max_length=16)
    resolution_evidence_ids: list[str] = Field(default_factory=list,max_length=16)
    reason: str = Field(min_length=1,max_length=1000)

class Decision(Contract):
    action: Literal['search','finish']
    requests: list[SearchRequest] = Field(default_factory=list,max_length=8)
    needs: list[NeedPatch] = Field(default_factory=list,max_length=12)
    conflicts: list[ConflictPatch] = Field(default_factory=list,max_length=8)
    reason: str = Field(min_length=1,max_length=1000)
    @model_validator(mode='after')
    def consistent(self):
        if (self.action=='search') != bool(self.requests): raise ValueError('search needs requests; finish forbids them')
        return self

def decision_schema(mode,limits):
    schema=Decision.model_json_schema()
    schema['properties']['requests']['maxItems']=limits.searches if mode=='static_batch' else 1
    if mode!='research':
        for name in ('needs','conflicts'):schema['properties'][name]['maxItems']=0
    return schema

class ResearchState:
    def __init__(self,question):
        self.root={'id':'root','question':question,'status':'open','parent_id':None,'reason':'original obligation','evidence_ids':[]}
        self.needs={'root':dict(self.root)};self.conflicts={};self.events=[]
    def update(self,decision,visible):
        import copy
        needs=copy.deepcopy(self.needs); conflicts=copy.deepcopy(self.conflicts); events=[]
        for p in decision.needs:
            value=p.model_dump();old=needs.get(p.id)
            if p.id=='root' and (p.question!=self.root['question'] or p.status=='superseded' or p.parent_id is not None):
                raise ScopeViolation('Original obligation cannot be removed or rewritten')
            if p.id!='root' and (p.parent_id not in needs or p.parent_id==p.id): raise ScopeViolation('Need requires existing parent')
            if old and old['parent_id']!=p.parent_id: raise ScopeViolation('Immutable need lineage')
            if any(e not in visible for e in p.evidence_ids): raise ScopeViolation('Unknown need evidence')
            if p.status=='supported' and not p.evidence_ids: raise ScopeViolation('Supported need requires evidence')
            events.append({'kind':'need','before':old,'after':value});needs[p.id]=value
        for p in decision.conflicts:
            value=p.model_dump();old=conflicts.get(p.id)
            if any(e not in visible for e in p.evidence_ids+p.resolution_evidence_ids): raise ScopeViolation('Unknown conflict evidence')
            if old and not set(old['evidence_ids'])<=set(p.evidence_ids): raise ScopeViolation('Original conflict evidence must survive')
            if p.status=='resolved' and (not old or not set(p.resolution_evidence_ids)-set(old['evidence_ids'])):
                raise ScopeViolation('Conflict resolution requires additional observed evidence')
            if not old and p.status!='open': raise ScopeViolation('New conflict starts open')
            events.append({'kind':'conflict','before':old,'after':value});conflicts[p.id]=value
        self.needs=needs;self.conflicts=conflicts;self.events+=events
    def public(self): return {'needs':list(self.needs.values()),'conflicts':list(self.conflicts.values()),'notes_are_evidence':False}
