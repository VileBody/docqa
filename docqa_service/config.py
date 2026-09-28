import os
import importlib.metadata
from pathlib import Path
from dataclasses import dataclass
from docqa_rag.types import Profile
from docqa_rag.util import read, digest

ROOT=Path(__file__).resolve().parents[1]

@dataclass
class Settings:
    data: Path
    redis_url: str
    qdrant_url: str
    api_key: str
    profile_path: Path
    namespace: str='docqa2'
    allow_pdf: bool=False
    test_mode: bool=False
    model_calls: bool=False
    ledger: str=''
    lease_seconds: int=60
    reader_profile: Path|None=None

    @classmethod
    def environment(cls):
        return cls(Path(os.getenv('DOCQA_DATA_DIR', str(ROOT/'var/stage2'))),
            os.getenv('DOCQA_REDIS_URL','redis://127.0.0.1:16379/0'),
            os.getenv('DOCQA_QDRANT_URL','http://127.0.0.1:16334'),
            os.getenv('DOCQA_API_KEY',''),Path(os.getenv('DOCQA_PROFILE',str(ROOT/'profiles/stage_2/txt.json'))),
            os.getenv('DOCQA_NAMESPACE','docqa2'),os.getenv('DOCQA_ALLOW_PDF')=='1',
            os.getenv('DOCQA_TEST_MODE')=='1',os.getenv('DOCQA_MODEL_CALLS_ENABLED')=='1',
            os.getenv('DOCQA_BUDGET_LEDGER',''),int(os.getenv('DOCQA_LEASE_SECONDS','60')),
            Path(os.environ['DOCQA_READER_PROFILE']) if os.getenv('DOCQA_READER_PROFILE') else None)

    def reader(self):
        if self.reader_profile is None:return None
        from .submission import Submission
        return Submission.load(self.reader_profile)

    def profile(self):
        data=read(self.profile_path)
        for env,field in {'DOCQA_RERANK_INPUT_LIMIT':'candidate_limit',
                'DOCQA_RERANK_OUTPUT_LIMIT':'rerank_limit','DOCQA_PACK_CHARS':'pack_chars',
                'DOCQA_CHUNK_SIZE':'chunk_size','DOCQA_CHUNK_OVERLAP':'chunk_overlap'}.items():
            if os.getenv(env):data[field]=int(os.environ[env])
        if os.getenv('DOCQA_FUSION'):data['fusion']=os.environ['DOCQA_FUSION']
        if os.getenv('DOCQA_RERANK_MODEL'):
            data['reranker']['model_id']=os.environ['DOCQA_RERANK_MODEL']
            revision=os.getenv('DOCQA_RERANK_REVISION')
            if not revision:raise ValueError('DOCQA_RERANK_REVISION required with model override')
            data['reranker']['revision']=revision
            data['reranker']['tokenizer_revision']=revision
        if os.getenv('DOCQA_RERANK_INSTRUCTION'):
            data['reranker']['query_instruction']=os.environ['DOCQA_RERANK_INSTRUCTION']
        for role,binding in [('EMBED',data['embedding']),('QWEN',data['generators']['qwen']),('OPENAI',data['generators']['openai'])]:
            if os.getenv('DOCQA_'+role+'_MODEL'):
                changed=os.environ['DOCQA_'+role+'_MODEL']!=binding['model_id']
                revision=os.getenv('DOCQA_'+role+'_REVISION') or (binding['revision'] if os.environ['DOCQA_'+role+'_MODEL']==binding['model_id'] else None)
                if not revision:raise ValueError('Explicit revision required with model override: '+role)
                binding.update(model_id=os.environ['DOCQA_'+role+'_MODEL'],revision=revision,tokenizer_revision=revision)
                # A changed checkpoint must supply its own compatible tokenizer.
                if changed:
                    binding['tokenizer_path']=None
                    binding['chat_template_sha256']=None
                if os.getenv('DOCQA_'+role+'_TOKENIZER_PATH'):
                    binding['tokenizer_path']=os.environ['DOCQA_'+role+'_TOKENIZER_PATH']
                    binding['chat_template_sha256']=None
            for suffix,field in [('INPUT_TOKENS','model_input_budget'),('CONTEXT_TOKENS','model_context_budget'),('OUTPUT_TOKENS','max_output_tokens')]:
                if os.getenv('DOCQA_'+role+'_'+suffix):binding[field]=int(os.environ['DOCQA_'+role+'_'+suffix])
        if os.getenv('DOCQA_EMBED_DIMENSION'):data['embedding']['dimension']=int(os.environ['DOCQA_EMBED_DIMENSION'])
        p=Profile.model_validate(data)
        if p.rerank_limit>p.candidate_limit:raise ValueError('Reranker output limit exceeds input candidate limit')
        selected=self.reader()
        if selected and selected.binding_sha256 and digest(p.generators[selected.generator].model_dump())!=selected.binding_sha256:
            raise ValueError('Reader binding differs from selected profile')
        if self.test_mode:
            p.embedding.dimension=8
            p.enforce_context_limits=False
        return p

    def pipeline(self):
        from docqa_rag.extraction import TEXT_PARSER, PARSER
        if self.reader_profile is not None:
            p=self.profile()
            # Reader/generation/retrieval settings cannot invalidate document vectors.
            return digest({'version':'submission-ingestion-v1','embedding':p.embedding.model_dump(),
                'chunk_size':p.chunk_size,'chunk_overlap':p.chunk_overlap,
                'txt_parser':TEXT_PARSER,'optional_pdf':PARSER if self.allow_pdf else False,
                'backend':'test_doubles' if self.test_mode else 'remote_models',
                'dependencies':{n:importlib.metadata.version(n) for n in ('qdrant-client','langchain-text-splitters')},
                'code':{n:digest((ROOT/n).read_bytes()) for n in ('docqa_rag/store.py','docqa_rag/extraction.py','docqa_rag/sparse.py')}})
        return digest({'version':'stage2-txt-atomic-v1','profile':self.profile().model_dump(),
            'txt_parser':TEXT_PARSER,'optional_pdf':PARSER if self.allow_pdf else False,
            'backend':'test_doubles' if self.test_mode else 'remote_models','critic_mode':'shadow',
            'dependencies':{n:importlib.metadata.version(n) for n in ('qdrant-client','langchain-text-splitters')},
            'code':{n:digest((ROOT/n).read_bytes()) for n in ('docqa_rag/store.py','docqa_rag/extraction.py','docqa_rag/sparse.py','docqa_service/core.py')}})
