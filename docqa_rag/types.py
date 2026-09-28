"""Transport contracts shared by library, CLI and future service."""
from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator

class Contract(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)

class Box(Contract):
    start: int
    end: int
    bbox: tuple[float, float, float, float]

class Page(Contract):
    page: int
    start: int
    end: int
    text: str
    boxes: list[Box]
    extraction_status: Literal['text', 'blank'] = 'text'

class Document(Contract):
    doc_id: str
    doc_title: str
    pdf_sha256: str
    text_sha256: str
    text: str
    pages: list[Page]
    parser: str
    assembly: Literal['single_file'] = 'single_file'

class Chunk(Contract):
    source_id: str
    doc_id: str
    doc_title: str
    chunk_index: int
    page_from: int
    page_to: int
    section: str | None = None
    text: str
    char_start: int
    char_end: int
    text_sha256: str
    pdf_sha256: str

class QueryPlan(Contract):
    raw: str
    lexical: str
    dense: str
    anchors: list[str]
    policy: Literal['raw', 'anchors']

class Hit(Contract):
    chunk: Chunk
    retrieval_score: float
    rerank_score: float | None = None

class Reference(Contract):
    source_id: str
    quote: str | None = Field(default=None, min_length=1)
    span_id: str | None = None
    start: int | None = Field(default=None, ge=0)
    end: int | None = Field(default=None, ge=1)
    occurrence: int | None = Field(default=None, ge=1)
    page: int | None = Field(default=None, ge=1)

class Claim(Contract):
    text: str = Field(min_length=1)
    references: list[Reference] = Field(min_length=1)

class Draft(Contract):
    status: Literal['complete', 'partial', 'ambiguous', 'contradictory', 'not_found']
    claims: list[Claim]
    missing: list[str]

class SpanReference(Contract):
    source_id: str
    span_id: str

class SpanClaim(Contract):
    text: str
    references: list[SpanReference] = Field(min_length=1)

class SpanDraft(Contract):
    status: Literal['complete', 'partial', 'ambiguous', 'contradictory', 'not_found']
    claims: list[SpanClaim]
    missing: list[str]

def _strict_answerability_state(draft):
    """Structural consistency only, never a semantic answerability judgment."""
    if draft.status=='complete' and (not draft.claims or draft.missing):
        raise ValueError('complete requires claims and empty missing')
    if draft.status=='partial' and (not draft.claims or not draft.missing):
        raise ValueError('partial requires claims and missing')
    if draft.status in {'not_found','ambiguous','contradictory'} and draft.claims:
        raise ValueError('unanswered status must not contain claims')
    if any(not item.strip() for item in draft.missing):
        raise ValueError('missing items must be nonempty')
    return draft

class StrictDraft(Draft):
    """Opt-in renderer policy; the legacy Draft wire schema remains unchanged."""
    @model_validator(mode='after')
    def consistent(self):
        return _strict_answerability_state(self)

class StrictSpanDraft(SpanDraft):
    """Post-parse validation, not a replacement response_format schema."""
    @model_validator(mode='after')
    def consistent(self):
        return _strict_answerability_state(self)

class SpanRangeReference(Contract):
    source_id: str
    start_span_id: str
    end_span_id: str

class SpanRangeClaim(Contract):
    text: str = Field(min_length=1)
    references: list[SpanRangeReference] = Field(min_length=1)

class SpanRangeDraft(Contract):
    status: Literal['complete', 'partial', 'ambiguous', 'contradictory', 'not_found']
    claims: list[SpanRangeClaim]
    missing: list[str]

class Citation(Contract):
    match_policy: Literal['exact', 'whitespace_runs', 'issued_span', 'explicit_range'] = 'exact'
    source_id: str
    document_id: str
    page: int
    text: str
    char_start: int
    char_end: int
    text_sha256: str
    pdf_sha256: str
    boxes: list[Box]

class Answer(Contract):
    answer: str
    found: bool
    citations: list[Citation]
    status: Literal['complete', 'partial', 'ambiguous', 'contradictory', 'not_found', 'unsupported']
    support_check: str
    @model_validator(mode='after')
    def valid(self):
        if self.found != bool(self.citations):
            raise ValueError('found must agree with resolved citations')
        return self

class Verdict(Contract):
    supported: bool
    reason: str

class Binding(Contract):
    family: Literal['openai', 'qwen']
    model_id: str
    revision: str = Field(min_length=1)
    tokenizer_revision: str = Field(min_length=1)
    endpoint_env: str
    key_env: str
    dimension: int | None = None
    document_instruction: str = ''
    query_instruction: str = 'Given a question, retrieve passages that answer it.'
    quantization: str = 'none'
    serving_image_digest: str | None = None
    chat_template_sha256: str | None = None
    reasoning: str = 'disabled'
    live_smoke: bool = False
    provider: str = 'unspecified'
    origin_verification: str = 'not_verified'
    model_input_budget: int = Field(default=16000, ge=128)
    model_context_budget: int = Field(default=32768, ge=128)
    max_output_tokens: int = Field(default=2048, ge=1, le=16384)
    tokenizer_path: str | None = None
    context_limit_source: str = 'unverified_local_cap'

    @model_validator(mode='after')
    def explicit_reasoning_reserve(self):
        from .chat_parameters import reasoning_effort
        effort=reasoning_effort(self)
        if self.max_output_tokens>2048 and not (
                self.family=='openai' and self.model_id.startswith(('gpt-6-astra','gpt-6-sol'))
                and effort not in {None,'none'}):
            raise ValueError('Output above 2048 requires an explicit Astra/Sol reasoning binding')
        return self

class Profile(Contract):
    embedding: Binding
    reranker: Binding
    generators: dict[str, Binding]
    chunk_size: int = Field(default=1200, ge=80, le=12000)
    chunk_overlap: int = Field(default=160, ge=0)
    candidate_limit: int = Field(default=24, ge=1, le=200)
    rerank_limit: int = Field(default=6, ge=1, le=50)
    rerank_threshold: float = Field(default=0.1, ge=0, le=1)
    pack_chars: int = Field(default=8000, ge=100, le=65536)
    pack_reference_tokens: int = Field(default=8192, ge=100, le=32768)
    fusion: Literal['rrf', 'dbsf'] = 'rrf'
    query_policy: Literal['raw', 'anchors'] = 'raw'
    index_cache_policy: Literal['auto', 'rebuild', 'persist_local'] = 'auto'
    reference_mode: Literal['quote', 'span', 'span_range'] = 'quote'
    threshold_enabled: bool = True
    enforce_context_limits: bool = False
    debug_artifacts: bool = False
    data_classification: Literal['synthetic', 'public', 'private'] = 'synthetic'
    debug_access_policy: str | None = None
    debug_retention_hours: int = Field(default=72, ge=1, le=168)
    max_validation_repairs: int = Field(default=0, ge=0, le=0)
    @model_validator(mode='after')
    def roles(self):
        if self.embedding.family != 'qwen' or self.reranker.family != 'qwen':
            raise ValueError('Qwen required for embedding/reranking')
        if set(self.generators) != {'openai','qwen'}:
            raise ValueError('both OpenAI/Qwen generator bindings required')
        if any(k != v.family for k,v in self.generators.items()):
            raise ValueError('generator family mismatch')
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError('overlap must be smaller than chunk')
        if self.debug_artifacts and self.data_classification=='private' and not self.debug_access_policy:
            raise ValueError('Private diagnostics require an explicit access/retention policy')
        return self
