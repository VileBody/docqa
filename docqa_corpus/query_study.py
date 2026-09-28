"""Offline contracts and dry-run plans for query/agent studies.

No model, Qdrant, HTTP, or agent runtime is implemented here. Controllers propose
read-only operations; the future application must enforce these contracts before
executing them. Python >= 3.10, standard library only.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from enum import Enum
from pathlib import Path
from typing import Any, Protocol
import hashlib
import json
import re

from .benchmark import Benchmark
from .core import Corpus


class ModelFamily(str, Enum):
    OPENAI = 'openai'
    QWEN = 'qwen'


class ExecutionMode(str, Enum):
    FLOW = 'flow'
    TOOL_ONCE = 'tool_once'
    REACTIVE = 'reactive'
    RESEARCH = 'research'
    STATIC_BATCH = 'static_batch'


def _positive_int(value: int, name: str) -> None:
    if type(value) is not int or value < 1:
        raise ValueError(f'{name} must be a positive integer')


def _text(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{name} must be nonempty text')


@dataclass(frozen=True)
class RequestScope:
    request_id: str
    document_id: str
    snapshot_id: str

    def __post_init__(self) -> None:
        for f in fields(self):
            _text(getattr(self, f.name), f.name)


@dataclass(frozen=True)
class RetrievalView:
    sparse_query: str
    dense_query: str
    purpose: str = 'answer_evidence'
    representation: str = 'query'

    def __post_init__(self) -> None:
        _text(self.sparse_query, 'sparse_query')
        _text(self.dense_query, 'dense_query')
        if self.representation not in {'query', 'expanded_query', 'hypothetical_document'}:
            raise ValueError('Unknown retrieval representation')
        if len(self.sparse_query) > 12000 or len(self.dense_query) > 12000:
            raise ValueError('Retrieval text exceeds contract limit')


@dataclass(frozen=True)
class QueryPlan:
    original_question: str
    views: tuple[RetrievalView, ...]
    anchors: tuple[str, ...] = ()
    policy: str = 'raw'
    revision: str = '1'

    def __post_init__(self) -> None:
        _text(self.original_question, 'original_question')
        if not isinstance(self.views, tuple) or not self.views or len(self.views) > 8:
            raise ValueError('views must be a tuple of 1..8 RetrievalView items')
        if not all(isinstance(v, RetrievalView) for v in self.views):
            raise TypeError('Unexpected view type')
        if not isinstance(self.anchors, tuple):
            raise TypeError('anchors must be immutable tuple')
        if self.original_question not in {v.sparse_query for v in self.views}:
            raise ValueError('An unchanged original sparse query is required')
        for anchor in self.anchors:
            _text(anchor, 'anchor')
            if anchor not in self.original_question:
                raise ValueError('User anchors must originate in the original question')

    @property
    def sha256(self) -> str:
        payload=json.dumps(asdict(self),ensure_ascii=False,sort_keys=True,separators=(',',':'))
        return hashlib.sha256(payload.encode()).hexdigest()


def raw_plan(question: str) -> QueryPlan:
    return QueryPlan(question,(RetrievalView(question,question),))


def anchor_plan(question: str) -> QueryPlan:
    """Conservative whitespace normalization only; never fixes codes or invents synonyms.

    A diagnostic extractor lists numeric/code-like spans. It is not an exhaustive
    entity detector. Literal original sparse text is retained regardless of extraction.
    """
    _text(question,'question')
    normalized=' '.join(question.split())
    anchors=tuple(dict.fromkeys(re.findall(r'(?<!\w)[\w]*\d[\w./%,-]*(?!\w)',question)))
    return QueryPlan(question,(RetrievalView(question,normalized),),anchors,'anchor')


@dataclass(frozen=True)
class EpisodeBudget:
    search_requests: int = 4
    query_views: int = 4
    rerank_pairs: int = 200
    observed_source_reference_tokens: int = 16384
    final_source_reference_tokens: int = 8192
    model_calls: int = 8
    generated_native_tokens: int = 8192
    deadline_seconds: int = 180

    def __post_init__(self) -> None:
        for f in fields(self):
            _positive_int(getattr(self,f.name),f.name)
        if self.final_source_reference_tokens > self.observed_source_reference_tokens:
            raise ValueError('Final evidence cannot exceed total exposed evidence budget')
        if self.search_requests > self.query_views:
            raise ValueError('Each retrieval request needs at least one query view')


@dataclass(frozen=True)
class Usage:
    search_requests: int = 0
    query_views: int = 0
    rerank_pairs: int = 0
    observed_source_reference_tokens: int = 0
    final_source_reference_tokens: int = 0
    model_calls: int = 0
    generated_native_tokens: int = 0
    elapsed_seconds: float = 0.0

    def __post_init__(self) -> None:
        for f in fields(self):
            v=getattr(self,f.name)
            if f.name=='elapsed_seconds':
                if isinstance(v,bool) or not isinstance(v,(int,float)) or not (0<=v<float('inf')):
                    raise ValueError('Invalid elapsed time')
            elif type(v) is not int or v<0:
                raise ValueError(f'Invalid usage: {f.name}')


def exceeded_limits(budget: EpisodeBudget, usage: Usage) -> tuple[str,...]:
    """Post-step accounting; the future runner must reserve/limit before calling tools."""
    result=[]
    for f in fields(budget):
        value=usage.elapsed_seconds if f.name=='deadline_seconds' else getattr(usage,f.name)
        if value>getattr(budget,f.name):result.append(f.name)
    return tuple(result)


@dataclass(frozen=True)
class SearchResultReceipt:
    scope: RequestScope
    search_id: str
    evidence_ids: tuple[str,...]
    reranked: bool
    source_kind: str = 'original_document'


def validate_receipt(scope: RequestScope, receipt: SearchResultReceipt) -> None:
    if receipt.scope!=scope:
        raise PermissionError('Wrong request/document/snapshot in tool response')
    _text(receipt.search_id,'search_id')
    if receipt.reranked is not True:
        raise ValueError('Only reranked evidence may be exposed to the model')
    if receipt.source_kind!='original_document':
        raise ValueError('Generated hypotheses and summaries are not source evidence')
    if not isinstance(receipt.evidence_ids,tuple) or len(receipt.evidence_ids)!=len(set(receipt.evidence_ids)):
        raise ValueError('Evidence IDs must be a unique tuple')
    for evidence_id in receipt.evidence_ids:_text(evidence_id,'evidence_id')


@dataclass(frozen=True)
class ControllerAction:
    kind: str
    query: str | None = None
    evidence_ids: tuple[str,...] = ()
    stop_reason: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in {'search','finalize'}:
            raise ValueError('Only search/finalize actions are allowed')
        if self.kind=='search':
            _text(self.query,'query')
            if self.stop_reason is not None:raise ValueError('Search action cannot stop')
        elif self.query is not None:
            raise ValueError('Finalize cannot submit a query')
        if self.kind=='finalize' and self.stop_reason not in {
            'evidence_collected','insufficient_observed_evidence','budget_exhausted',
            'tool_error','ambiguous_question','invalid_input','conflict_detected'}:
            raise ValueError('Explicit stop reason required')


class ResearchController(Protocol):
    def next_action(self, *, question: str, observations: tuple[dict[str,Any],...],
                    needs: tuple[dict[str,Any],...], budget: EpisodeBudget,
                    usage: Usage) -> ControllerAction:
        """Contract only. No access to gold, audit labels, files outside the scope or prior requests."""
        ...


def validate_study_config(config: dict[str,Any]) -> None:
    for role in ('generator_family','controller_family','planner_family'):
        v=config.get(role)
        if v is not None:ModelFamily(v)
    mode=ExecutionMode(config.get('mode','flow'))
    if config.get('external_search',False):raise ValueError('External search is outside this study')
    if config.get('hosted_file_search',False):raise ValueError('Hosted search would replace Qdrant')
    if config.get('cross_request_memory',False):raise ValueError('Cross-question memory forbidden')
    if config.get('stage')!='query_screen' and config.get('generator_family') is None:
        raise ValueError('Final generator must be OpenAI or Qwen')
    budget=EpisodeBudget(**config.get('budget',{}))
    if mode in (ExecutionMode.FLOW,ExecutionMode.TOOL_ONCE) and budget.search_requests!=1:
        raise ValueError('Single-retrieval controls must use one search request')
    if mode in (ExecutionMode.TOOL_ONCE,ExecutionMode.REACTIVE,ExecutionMode.RESEARCH,ExecutionMode.STATIC_BATCH):
        if config.get('controller_family') is None:raise ValueError('Controller family required')


def make_plan(root: str|Path) -> dict[str,Any]:
    """Build 32 frozen *plans*: 14 query screens, 10 end-to-end arms, 8 replays."""
    corpus=Corpus(root);benchmark=Benchmark(corpus)
    if not corpus.verify()['ok']:raise ValueError('Corpus integrity check failed')
    rows=benchmark.questions('dev')
    specs=json.loads((Path(root)/'experiments/query_agents/query_policies.json').read_text())
    fingerprint=benchmark.fingerprint()
    jobs=[]
    for p in specs:
        for provider in ([None] if p['provider']=='none' else list(ModelFamily)):
            fam=provider.value if provider else None
            budget=EpisodeBudget(search_requests=1,query_views=p['views'])
            jobs.append(dict(id=f'QPS-{p["id"]}-{fam or "none"}',stage='query_screen',mode='flow',
                query_policy=p['id'],planner_family=fam,controller_family=None,generator_family=None,
                budget=asdict(budget),answer_generations_per_question=0,
                planner_calls_per_question=0 if fam is None else 1,
                reuses=None,question_count=len(rows),repeats=1))
    for mode in ExecutionMode:
        for family in ModelFamily:
            single=mode in (ExecutionMode.FLOW,ExecutionMode.TOOL_ONCE)
            budget=EpisodeBudget(search_requests=1 if single or mode==ExecutionMode.STATIC_BATCH else 4,
                                 query_views=1 if single else 4)
            jobs.append(dict(id=f'AG-{mode.value}-{family.value}',stage='mode_comparison',mode=mode.value,
                query_policy='raw' if mode==ExecutionMode.FLOW else 'controller_generated_no_second_rewriter',
                planner_family=None,controller_family=None if mode==ExecutionMode.FLOW else family.value,
                generator_family=family.value,budget=asdict(budget),answer_generations_per_question=1,
                controller_model_calls='variable_within_budget; includes planning/critique/repairs',
                reuses=None,question_count=len(rows),repeats=1))
    for mode in (ExecutionMode.REACTIVE,ExecutionMode.RESEARCH):
        for controller in ModelFamily:
            for generator in ModelFamily:
                jobs.append(dict(id=f'RP-{mode.value}-{controller.value}-to-{generator.value}',
                    stage='evidence_replay',mode=mode.value,query_policy='frozen_source_pack',
                    planner_family=None,controller_family=controller.value,generator_family=generator.value,
                    budget=asdict(EpisodeBudget()),answer_generations_per_question=1,
                    reuses=f'AG-{mode.value}-{controller.value}',question_count=len(rows),repeats=1,
                    note='Offline synth replay only; reuse same source bytes/order for both generators. No new retrieval or controller calls.'))
    for job in jobs:
        validate_study_config(job)
        job.update(status='planned_not_executed',requires_runtime_adapter=True,
                   dataset_hash=fingerprint,gold_version=corpus.manifest['gold_version'],
                   view='dev',question_ids=[q['id'] for q in rows],external_search=False,
                   hosted_file_search=False,cross_request_memory=False)
    return dict(schema_version='1.0',status='planned_not_executed',
        note='Plans and contract checks only. Pin models and implement adapters before inference. No automatic Cartesian sweep.',
        allowed_generator_families=[f.value for f in ModelFamily],
        dataset_hash=fingerprint,questions_per_job=len(rows),root_families=len({q['family_id'] for q in rows}),
        phases=['query_screen','mode_comparison','evidence_replay'],jobs=jobs,
        job_count=len(jobs),screen_planner_calls=sum(j.get('planner_calls_per_question',0)*len(rows) for j in jobs),
        planned_final_answer_calls=sum(j['answer_generations_per_question']*len(rows) for j in jobs),
        costs_not_included=['embedding','reranking','variable controller calls','judges','retries','latency','provider reasoning tokens','new sensitivity runs'],
        comparison='Family-macro paired on base documents. M/C/paraphrases separate; real corpus excluded until approved.')
