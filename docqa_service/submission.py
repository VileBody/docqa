"""Explicit product wiring; historical profiles retain their original behavior."""
from pathlib import Path
from typing import Literal
from pydantic import Field
from docqa_rag.types import Contract
from docqa_rag.reader_control import ReaderPrompt, ReaderAdapter, request_parts
from docqa_rag.util import read, digest
from docqa_rag.planning import PlannedFlow
from .config import ROOT

class Submission(Contract):
    version: str
    generator: Literal['openai']
    prompt_config: str
    answerability_policy: Literal['strict_answerability_v1']
    critic_mode: Literal['off','shadow']
    query_policy: Literal['raw']
    generation_cache: Literal['disabled']
    retrieval_profile: str
    origin: str
    request_timeout_s: int=Field(default=60,ge=1,le=600)
    max_attempts: int=Field(default=3,ge=1,le=3)
    prices_path: str="profiles/stage_1/prices.json"
    binding_sha256: str|None=None

    @classmethod
    def load(cls, path):
        obj=cls.model_validate(read(path))
        cfg=obj.prompt()
        if cfg.prompt!='P1' or cfg.public_policy!=obj.answerability_policy:
            raise ValueError('Submission requires selected P1/strict policy')
        # Validate assets and all demonstrations before any paid call.
        parts=request_parts('Startup integrity check',[],cfg,ROOT)
        if len(parts[4])!=12:raise ValueError('Exactly six teaching pairs required')
        return obj

    def prompt(self):return ReaderPrompt.model_validate(read(ROOT/self.prompt_config))

    def adapter(self, profile, calls):
        b=profile.generators[self.generator]
        if self.binding_sha256 and digest(b.model_dump())!=self.binding_sha256:
            raise ValueError("Reader binding differs from selected profile")
        return ReaderAdapter(b,calls,max_tokens=b.max_output_tokens,
                             prompt_config=self.prompt(),workspace_root=ROOT,request_timeout_s=self.request_timeout_s)


def make_flow(settings,index,embedding,reranker,generator,*,query_policy='raw',planner=None):
    selected=settings.reader()
    return PlannedFlow(index,embedding,reranker,generator,settings.profile(),
        generator,critic_mode=selected.critic_mode if selected else 'shadow',
        answerability_policy=selected.answerability_policy if selected else 'legacy',
        query_policy=query_policy,planner=planner)
