"""Explicit production adapters; deterministic doubles only with TEST_MODE=1."""
import hashlib, math
from pathlib import Path
from docqa_rag.types import Draft, Claim, Reference, Verdict
from docqa_rag.util import read, ModelUnavailable
from docqa_rag.budget import Budget
from docqa_rag.adapters import Calls, QwenEmbeddings, QwenReranker, ChatAdapter
from .config import ROOT
from docqa_rag.agent_loop.protocol import InstructionReranker

class TestEmbedding:
    def embed_documents(self,texts):return [self.embed_query(t) for t in texts]
    def embed_query(self,text):
        values=[b/255+.01 for b in hashlib.sha256(text.encode()).digest()[:8]]
        norm=math.sqrt(sum(x*x for x in values));return [x/norm for x in values]
class TestReranker:
    def rank(self,question,hits):return [h.model_copy(update={'rerank_score':.9}) for h in hits]
class TestGenerator:
    def generate(self,question,hits):
        if not hits:return Draft(status='not_found',claims=[],missing=[])
        h=hits[0];return Draft(status='partial',claims=[Claim(text=h.chunk.text,references=[Reference(source_id=h.chunk.source_id,quote=h.chunk.text,start=0,end=len(h.chunk.text))])],missing=['Test double; no semantic inference'])
    def check(self,claim,quotes):return Verdict(supported=False,reason='Deliberate shadow rejection in infrastructure test')

def calls(settings):
    if not settings.model_calls or not settings.ledger or not Path(settings.ledger).is_file():
        raise ModelUnavailable('Remote model calls disabled or authorized ledger missing')
    state=read(settings.ledger);budget=Budget(settings.ledger,state['max_usd'],state['max_calls'])
    selected=settings.reader()
    return Calls(budget,read(ROOT/(selected.prices_path if selected else 'profiles/stage_1/prices.json'))['prices'],max_attempts=selected.max_attempts if selected else 3)

def embedding(settings):
    return TestEmbedding() if settings.test_mode else QwenEmbeddings(settings.profile().embedding,calls(settings))

def create(settings,generator=None):
    selected=settings.reader()
    generator=generator or (selected.generator if selected else 'qwen')
    if selected and generator!=selected.generator:raise ValueError('Generator differs from submission profile')
    if settings.test_mode:return TestEmbedding(),TestReranker(),TestGenerator()
    c=calls(settings);p=settings.profile()
    return QwenEmbeddings(p.embedding,c),InstructionReranker(p.reranker,c),selected.adapter(p,c) if selected else ChatAdapter(p.generators[generator],c,max_tokens=p.generators[generator].max_output_tokens,reference_mode=p.reference_mode,enforce_context_limits=p.enforce_context_limits)
