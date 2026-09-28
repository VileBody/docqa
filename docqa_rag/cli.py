from __future__ import annotations
import argparse,json,os,shutil
from pathlib import Path
from dotenv import load_dotenv
from .types import Profile
from .util import read,write,digest
from .store import Index,index_spec,runner_scope,cache_decision
from .extraction import extract_source,split_document
from .budget import Budget
from .adapters import Calls,QwenEmbeddings,QwenReranker,ChatAdapter
from .flow import Flow,ExtractiveCritic
from .runner import Runner,Job,load_questions

DEFAULT_PROFILE=Path(__file__).resolve().parents[1]/'profiles/stage_1_1/default.json'

def load_profile(path):
    profile=Profile.model_validate(read(path))
    model=os.environ.get('DOCQA_OPENAI_MODEL')
    if model and model!=profile.generators['openai'].model_id:
        profile.generators['openai']=profile.generators['openai'].model_copy(update={
            'model_id':model,'revision':'provider-alias-immutable-revision-unverified','live_smoke':False})
    return profile

def adapters(profile,calls,generator='qwen',critic=False):
    embedding=QwenEmbeddings(profile.embedding,calls)
    reranker=QwenReranker(profile.reranker,calls)
    gen=ChatAdapter(profile.generators[generator],calls,reference_mode=profile.reference_mode,enforce_context_limits=profile.enforce_context_limits)
    check=gen if critic else None
    return embedding,reranker,gen,check

def main(argv=None):
    parser=argparse.ArgumentParser(description='PDF -> native Qdrant hybrid -> Qwen reranker -> cited flow answer')
    parser.add_argument('--env-file',default='.env'); parser.add_argument('--profile',default=str(DEFAULT_PROFILE))
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('inspect'); p.add_argument('runner'); p.add_argument('--output',required=True)
    p=sub.add_parser('dry-run'); p.add_argument('runner'); p.add_argument('--jobs',required=True); p.add_argument('--output',required=True)
    p=sub.add_parser('run'); p.add_argument('runner'); p.add_argument('--jobs',required=True); p.add_argument('--output',required=True)
    p.add_argument('--prices',required=True); p.add_argument('--budget-ledger',required=True); p.add_argument('--budget-usd',type=float,required=True)
    p.add_argument('--max-calls',type=int,default=200); p.add_argument('--deadline-s',type=int,default=1800)
    p.add_argument('--restore'); p.add_argument('--save-index'); p.add_argument('--question-limit',type=int,default=8)
    args=parser.parse_args(argv); load_dotenv(args.env_file,override=False); profile=load_profile(args.profile)
    manifest,docs=runner_scope(args.runner)
    if args.command=='inspect':
        report=[]
        for item in docs:
            try:
                doc=extract_source(Path(args.runner)/item['file'],item['id'],item['sha256']); chunks=split_document(doc,profile.chunk_size,profile.chunk_overlap)
                report.append({'doc_id':doc.doc_id,'status':'ready','pages':len(doc.pages),'chars':len(doc.text),'chunks':len(chunks),'text_sha256':doc.text_sha256})
            except Exception as e: report.append({'doc_id':item['id'],'status':'rejected','error_type':type(e).__name__,'reason':str(e)})
        write(args.output,{'dataset_hash':manifest['dataset_hash'],'documents':report}); print(json.dumps({'documents':len(report),'rejected':sum(r['status']=='rejected' for r in report)})); return 0
    job_plan=read(args.jobs)
    if job_plan['dataset_hash']!=manifest['dataset_hash']: raise ValueError('Job dataset hash mismatch')
    jobs=[Job.model_validate(j) for j in job_plan['jobs']]
    if args.command=='dry-run':
        qs=load_questions(args.runner); chosen=job_plan.get('question_ids') or [q['id'] for q in qs[:8]]
        write(args.output,{'status':'dry_run','dataset_hash':manifest['dataset_hash'],'index_fingerprint':digest(index_spec(args.runner,profile)),
            'jobs':[j.model_dump() for j in jobs],'question_ids':chosen,'model_calls_executed':0,
            'budget_status':'requires conservative price ceilings and persistent shared ledger',
            'external_resources':'not_owned_never_deleted','index_policy':profile.index_cache_policy})
        print(args.output); return 0
    if args.budget_usd<=0: raise ValueError('Positive explicitly authorized budget required for live calls')
    budget=Budget(args.budget_ledger,args.budget_usd,args.max_calls,args.deadline_s)
    calls=Calls(budget,read(args.prices)['prices']); embedding=QwenEmbeddings(profile.embedding,calls)
    index=None
    try:
        # Smoke before indexing; validate every requested generator separately.
        smoke={}
        for family in sorted({j.generator for j in jobs if j.generator}): smoke[family]=ChatAdapter(profile.generators[family],calls,reference_mode=profile.reference_mode,enforce_context_limits=profile.enforce_context_limits).smoke()
        write(Path(args.output)/'adapter_smoke.json',smoke)
        if args.restore: index=Index.restore(args.restore,digest(index_spec(args.runner,profile)))
        else: index=Index.build(args.runner,profile,embedding)
        def factory(job):
            reranker=QwenReranker(profile.reranker,calls)
            generator=ChatAdapter(profile.generators[job.generator],calls,reference_mode=profile.reference_mode,enforce_context_limits=profile.enforce_context_limits) if job.generator else None
            return Flow(index,embedding,reranker,generator,profile,generator if job.critic else None)
        runner=Runner(args.runner,args.output,index,profile,factory)
        ids=job_plan.get('question_ids') or [q['id'] for q in runner.questions[:args.question_limit]]
        runner.run(jobs,ids)
        if args.save_index: write(Path(args.output)/'index_export.json',index.export(args.save_index))
        write(Path(args.output)/'build_metrics.json',index.metrics)
        write(Path(args.output)/'calls.json',calls.records)
        return 0 if read(Path(args.output)/'run_manifest.json')['status']=='complete' else 2
    finally:
        if index: index.close()
