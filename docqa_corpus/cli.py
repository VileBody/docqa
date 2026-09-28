from __future__ import annotations
import argparse
import json
from pathlib import Path
from .core import Corpus
from .util import write_json,read_json


def main(argv:list[str]|None=None)->int:
    parser=argparse.ArgumentParser(prog='docqa-corpus',description='Offline Russian document-QA corpus and mutation kit')
    parser.add_argument('--root',type=Path,help='Kit or corpus directory; default adjacent corpus')
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('list');p.add_argument('--suite',choices=['S','M','R','C']);p.add_argument('--split',choices=['dev','holdout']);p.add_argument('--all',action='store_true',help='Also list missing official originals')
    p=sub.add_parser('verify');p.add_argument('--require-public',action='store_true');p.add_argument('--report',type=Path)
    p=sub.add_parser('export-questions');p.add_argument('output',type=Path);p.add_argument('--suite',choices=['S','M','R','C']);p.add_argument('--split',choices=['dev','holdout']);p.add_argument('--with-gold',action='store_true');p.add_argument('--include-mutations',action='store_true')
    p=sub.add_parser('mutate');p.add_argument('recipe',choices=[f'M{i:02d}' for i in range(1,9)]);p.add_argument('--params',default='{}',help='JSON object with named method parameters');p.add_argument('--out',type=Path);p.add_argument('--id',dest='document_id');p.add_argument('--overwrite',action='store_true')
    p=sub.add_parser('fetch-public');p.add_argument('ids',nargs='*');p.add_argument('--attempts',type=int,default=3);p.add_argument('--timeout',type=float,default=30)
    p=sub.add_parser('import-public');p.add_argument('document_id');p.add_argument('--file',type=Path,required=True)
    p=sub.add_parser('public-status');p.add_argument('--directory',type=Path);p.add_argument('--report',type=Path)
    p=sub.add_parser('inspect-public');p.add_argument('document_id');p.add_argument('--file',type=Path,required=True);p.add_argument('--report',type=Path)
    p=sub.add_parser('import-public-dir');p.add_argument('--directory',type=Path);p.add_argument('--ids',nargs='+');p.add_argument('--reviews',type=Path);p.add_argument('--require-all',action='store_true');p.add_argument('--report',type=Path)
    p=sub.add_parser('review-template');p.add_argument('document_id');p.add_argument('output',type=Path)
    p=sub.add_parser('apply-review');p.add_argument('review',type=Path)
    sub.add_parser('check-fixtures');sub.add_parser('build-fixtures');sub.add_parser('rebuild')
    p=sub.add_parser('preflight');p.add_argument('file',type=Path)
    p=sub.add_parser('audit-corpus');p.add_argument('--report',type=Path)
    p=sub.add_parser('export-runner');p.add_argument('output',type=Path);p.add_argument('--view',default='dev');p.add_argument('--allow-audit',action='store_true');p.add_argument('--with-evaluator',action='store_true')
    p=sub.add_parser('review-queue');p.add_argument('output',type=Path);p.add_argument('--view',default='dev');p.add_argument('--allow-audit',action='store_true')
    p=sub.add_parser('query-variants');p.add_argument('output',type=Path);p.add_argument('--view',default='dev');p.add_argument('--allow-audit',action='store_true')
    args=parser.parse_args(argv)
    try:
        corpus=Corpus(args.root);command=args.command
        if command in ('public-status','inspect-public','import-public-dir'):
            from .acquisition import PublicAcquisition
            acquisition=PublicAcquisition(corpus)
            if command=='public-status':result=acquisition.status(args.directory)
            elif command=='inspect-public':result=acquisition.inspect_file(args.document_id,args.file)
            else:result=acquisition.import_directory(args.directory,ids=args.ids,reviews=args.reviews,require_all=args.require_all)
            if args.report:write_json(args.report,result)
        elif command=='audit-corpus':
            from .curation import coverage
            from .benchmark import Benchmark
            result={'integrity':corpus.verify(),'coverage':coverage(corpus),'dataset_hash':Benchmark(corpus).fingerprint()}
            if args.report:write_json(args.report,result)
        elif command=='export-runner':
            from .benchmark import Benchmark
            result=Benchmark(corpus).export_runner(args.output,view=args.view,allow_audit=args.allow_audit,include_evaluator=args.with_evaluator)
        elif command=='review-queue':
            from .benchmark import Benchmark
            from .util import write_jsonl
            rows=Benchmark(corpus).review_queue(args.view,allow_audit=args.allow_audit);write_jsonl(args.output,rows)
            result={'output':str(args.output),'questions':len(rows),'status':'pending_review_not_a_certificate'}
        elif command=='query-variants':
            from .sensitivity import SensitivitySuite
            from .util import write_jsonl
            rows=SensitivitySuite(corpus).query_variants(args.view,allow_audit=args.allow_audit);write_jsonl(args.output,rows)
            result={'output':str(args.output),'variants':len(rows),'warning':'Evaluator fixtures include lineage; not a sanitized runner export.'}
        elif command=='list':
            result=[{**d.metadata,'available':d.available} for d in corpus.documents(suite=args.suite,split=args.split,available_only=not args.all)]
        elif command=='verify':
            result=corpus.verify(require_public=args.require_public)
            if args.report:write_json(args.report,result)
        elif command=='export-questions':result={'exported':corpus.export_questions(args.output,with_gold=args.with_gold,suite=args.suite,split=args.split,include_mutations=args.include_mutations),'output':str(args.output)}
        elif command=='mutate':
            from .mutations import MutationGenerator
            params=json.loads(args.params)
            if not isinstance(params,dict):raise ValueError('--params must be a JSON object')
            result=MutationGenerator(corpus).generate(args.recipe,output_dir=args.out,document_id=args.document_id,overwrite=args.overwrite,**params).metadata
        elif command=='fetch-public':
            from .public import fetch_public
            result=fetch_public(corpus,ids=args.ids,attempts=args.attempts,timeout=args.timeout)
        elif command=='import-public':
            from .public import import_public_file
            result=import_public_file(corpus,args.document_id,args.file)
        elif command=='review-template':
            from .public import review_template
            review_template(corpus,args.document_id,args.output);result={'output':str(args.output),'status':'requires_actual_review'}
        elif command=='apply-review':
            from .public import apply_review
            result=apply_review(corpus,args.review)
        elif command in ('check-fixtures','build-fixtures','preflight'):
            from .fixtures import verify_fixtures,build_fixtures,preflight
            result=(verify_fixtures(corpus) if command=='check-fixtures' else
                    {'fixtures':build_fixtures(corpus)} if command=='build-fixtures' else preflight(args.file))
        elif command=='rebuild':
            from .build import build_bundle,register
            from .mutations import MutationGenerator
            built=[]
            for d in list(corpus.documents(suite='S')):
                item=build_bundle(read_json(d.source_dir/'source.json'),read_json(d.source_dir/'qa_spec.json'),corpus.root)
                register(corpus.root,item);built.append(item['id'])
            built += [d.id for d in MutationGenerator(corpus).build_all(overwrite=True)]
            for d in list(corpus.documents(suite='C')):
                item=build_bundle(read_json(d.source_dir/'source.json'),read_json(d.source_dir/'qa_spec.json'),corpus.root)
                register(corpus.root,item);built.append(item['id'])
            from .curation import curate
            from .sensitivity import build_query_variants
            curate(corpus);build_query_variants(corpus)
            result={'rebuilt':built,'note':'Fixtures have separate hashes; run build-fixtures after changing parents.'}
        print(json.dumps(result,ensure_ascii=False,indent=2))
        return 1 if isinstance(result,dict) and result.get('ok') is False else 0
    except (ValueError,TypeError,KeyError,FileNotFoundError,FileExistsError,PermissionError,ImportError) as exc:
        parser.exit(2,f'{type(exc).__name__}: {exc}\nFor rendering/preflight: pip install -e ".[all]"\n')
    return 0

if __name__=='__main__':raise SystemExit(main())
