import argparse,json
from .config import Settings
from .core import Service

def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    u=sub.add_parser('upload');u.add_argument('file');u.add_argument('--classification',choices=['public','synthetic'],required=True)
    w=sub.add_parser('process');w.add_argument('task_id')
    retry=sub.add_parser('retry');retry.add_argument('task_id')
    q=sub.add_parser('ask');q.add_argument('document_id');q.add_argument('question');q.add_argument('--diagnostics',action='store_true')
    e=sub.add_parser('export');e.add_argument('destination')
    r=sub.add_parser('restore');r.add_argument('source');r.add_argument('--fingerprint',required=True)
    sub.add_parser('list');a=p.parse_args();service=Service(Settings.environment())
    if a.command=='upload':
        from pathlib import Path
        f=Path(a.file);result=service.upload(f.read_bytes(),f.name,a.classification)
    elif a.command=='process':result=service.process(a.task_id)
    elif a.command=='retry':result=service.retry(a.task_id)
    elif a.command=='ask':result=service.answer(a.document_id,a.question,diagnostics=a.diagnostics)
    elif a.command=='export':result=service.export(a.destination)
    elif a.command=='restore':result=service.restore(a.source,a.fingerprint)
    else:result=service.list_documents()
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
