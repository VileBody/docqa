"""Jobs execute locally; each completed item is atomically checkpointed before next call.

No imports from docqa_corpus; gold is only accepted by the separate evaluator.
"""
from pathlib import Path
import os,time
from pydantic import Field
from .types import Contract, Answer, Hit
from .store import runner_scope
from .util import digest, read, write, BudgetExceeded
from .adapters import generation_identity

class Job(Contract):
    id: str = Field(pattern=r'^[A-Za-z0-9_-]+$')
    origin_id: str
    branch: str = 'hybrid'
    reranker: bool = True
    generator: str | None = 'openai'
    critic: bool = False


def load_questions(runner):
    import json
    manifest,docs=runner_scope(runner); docids={d['id'] for d in docs}; qs=[]; ids=set()
    for line in (Path(runner)/'questions.jsonl').read_text().splitlines():
        q=json.loads(line)
        if set(q)!={'id','document_id','question'} or q['document_id'] not in docids or q['id'] in ids: raise ValueError('Unsafe question contract')
        ids.add(q['id']); qs.append(q)
    if len(qs)!=manifest['question_count']: raise ValueError('Question count mismatch')
    return qs

def code_fingerprint():
    return digest({p.name:digest(p.read_bytes()) for p in Path(__file__).parent.glob('*.py')})

class Runner:
    def __init__(self,runner,run_dir,index,profile,flow_factory):
        self.runner=Path(runner); self.output=Path(run_dir); self.index=index; self.profile=profile; self.flow_factory=flow_factory
        self.questions=load_questions(runner)
        if runner_scope(runner)[0]['dataset_hash']!=index.spec['dataset_hash']: raise ValueError('Dataset mismatch')
        self.signature={'dataset_hash':index.spec['dataset_hash'],'index_fingerprint':index.fingerprint,
            'runner_hash':digest({p.name:digest(p.read_bytes()) for p in self.runner.glob('*.json*')}),
            'profile':profile.model_dump(),'code_hash':code_fingerprint(),'generation_contract':generation_identity()}
    def dry_run(self,jobs,question_ids=None):
        qs=self.questions if question_ids is None else [q for q in self.questions if q['id'] in question_ids]
        if question_ids is not None and len(qs)!=len(set(question_ids)): raise ValueError('Unknown/duplicate question IDs')
        calls_per=sum((j.branch!='bm25')+j.reranker+(j.generator is not None)+(j.critic*self.profile.rerank_limit) for j in jobs)
        return {**self.signature,'jobs':[j.model_dump() for j in jobs],'question_ids':[q['id'] for q in qs],
            'max_requests_excluding_build_and_retries':len(qs)*calls_per,'retry_multiplier':3,
            'build_requests':'ceil(chunks/16), reserve separately','status':'dry_run_no_inference'}
    def run(self,jobs,question_ids=None,resume=True):
        plan=self.dry_run(jobs,question_ids); self.output.mkdir(parents=True,exist_ok=True)
        manifest_path=self.output/'run_manifest.json'; plan_hash=digest(plan)
        if manifest_path.exists() and read(manifest_path)['plan_hash']!=plan_hash: raise ValueError('Run config changed; use new run directory')
        write(manifest_path,{'plan':plan,'plan_hash':plan_hash,'status':'running'})
        selected=set(plan['question_ids']); results=[]
        # One runner owns a run folder. A crashed lock is explicit, not guessed stale.
        lock=self.output/'.lock'; fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
        try:
            for job in jobs:
                flow=self.flow_factory(job)
                for q in self.questions:
                    if q['id'] not in selected: continue
                    key=digest([self.signature,job.model_dump(),q])
                    result_path=self.output/'items'/(key+'.json')
                    if resume and result_path.exists():
                        cached=read(result_path)
                        if cached['key']!=key: raise ValueError('Checkpoint key mismatch')
                        if cached['status']=='success':
                            results.append({**cached,'cache_hit':True,'independent_sample':False}); continue
                    started=time.perf_counter(); trace=None
                    row={'key':key,'question_id':q['id'],'document_id':q['document_id'],'job':job.model_dump(),
                        'dataset_hash':self.signature['dataset_hash'],'cache_hit':False,'independent_sample':True}
                    from .debug import DebugReceipt
                    flow.debug=DebugReceipt(self.output/'debug'/(key+'.json'),self.profile,
                        {'question_id':q['id'],'job':job.model_dump(),'profile_hash':digest(self.profile.model_dump()),'bindings':{k:b.model_dump() for k,b in self.profile.generators.items()}})
                    try:
                        if job.generator is None:
                            _,trace=flow.retrieve(q['document_id'],q['question'],job.branch,job.reranker); answer=None
                        else:
                            # Freeze real retrieval independently of generator/critic for paired replay.
                            retrieval_profile=self.profile.model_dump(exclude={'generators'})
                            pack_key=digest([self.index.fingerprint,retrieval_profile,q,job.branch,job.reranker,self.signature['code_hash']])
                            pack_path=self.output/'packs'/(pack_key+'.json')
                            if pack_path.exists():
                                trace=read(pack_path); pack=[Hit.model_validate(h) for h in trace['pack']]
                                trace['retrieval_cache_hit']=True
                            else:
                                pack,trace=flow.retrieve(q['document_id'],q['question'],job.branch,job.reranker)
                                trace['retrieval_cache_hit']=False; write(pack_path,trace)
                            answer,details=flow.answer_pack(q['document_id'],q['question'],pack)
                            trace.update(details);trace['answer']=answer.model_dump()
                        row.update(status='success',answer=answer.model_dump() if answer else None,trace=trace)
                    except Exception as e:
                        row.update(status='budget_exhausted' if isinstance(e,BudgetExceeded) else 'error',error_type=type(e).__name__,error_message=str(e)[:400])
                        if trace is not None:row['trace']=trace
                    row['elapsed_s']=time.perf_counter()-started
                    write(result_path,row); results.append(row)
                    write(self.output/'results.json',results)
                    if row['status']=='budget_exhausted': return results
            return results
        finally:
            write(self.output/'results.json',results)
            write(manifest_path,{'plan':plan,'plan_hash':plan_hash,'status':'complete' if len(results)==len(jobs)*len(selected) and all(r['status']=='success' for r in results) else 'partial',
                                'results_sha256':digest((self.output/'results.json').read_bytes())})
            os.close(fd); lock.unlink()
