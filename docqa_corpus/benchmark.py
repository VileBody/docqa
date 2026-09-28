"""Named frozen evaluation views and leak-resistant runner exports (stdlib)."""
from __future__ import annotations
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any
from .core import Corpus
from .util import read_json, write_json, write_jsonl, sha256, safe_id


class Benchmark:
    def __init__(self, corpus: Corpus | str | Path | None=None):
        self.corpus=corpus if isinstance(corpus,Corpus) else Corpus(corpus)
        self.registry=read_json(self.corpus.root/'views.json')

    @property
    def views(self)->tuple[str,...]:
        return tuple(self.registry['views'])

    def fingerprint(self)->str:
        """Hashes actual dataset bytes, including annotations and view definitions."""
        h=hashlib.sha256()
        files=[self.corpus.root/'manifest.json', self.corpus.root/'views.json']
        for d in self.corpus.documents(available_only=False):
            files += [d.pdf_path, self.corpus.root/d.metadata['annotations'],
                      d.source_dir/'canonical.txt']
        sensitivity=self.corpus.root/'sensitivity'
        if sensitivity.exists(): files += sorted(sensitivity.glob('*.json'))+sorted(sensitivity.glob('*.jsonl'))
        for p in sorted(set(files)):
            rel=str(p.relative_to(self.corpus.root)).replace('\\','/')
            digest=sha256(p) if p.is_file() else 'MISSING'
            h.update((rel+'\0'+digest+'\n').encode('utf-8'))
        return h.hexdigest()

    def questions(self, view:str='dev', *, allow_audit:bool=False)->list[dict]:
        if view not in self.registry['views']:raise ValueError(f'Unknown view {view!r}; choose {self.views}')
        spec=self.registry['views'][view]
        if spec.get('requires_explicit_opt_in') and not allow_audit:
            raise PermissionError('Frozen audit view: pass allow_audit=True only after freezing system configurations.')
        rows=[]
        for i in spec['documents']:
            d=self.corpus.get(i)
            if not d.available:raise FileNotFoundError(f'{i}: original is not present; cannot score {view}')
            all_rows=d.questions(ready_only=False)
            valid=d.questions()
            if len(valid)!=len(all_rows) or not valid:
                raise ValueError(f'{i}: incomplete/unapproved annotations')
            if view=='public' and not all(q.get('review',{}).get('independent_human_review') is True for q in valid):
                raise ValueError(f'{i}: independent human review is required for the public comparison')
            rows.extend(valid)
        ids=[q['id'] for q in rows]
        if len(set(ids))!=len(ids):raise ValueError('Duplicate question IDs')
        return rows

    def export_runner(self, output:str|Path, *, view:str='dev', allow_audit:bool=False,
                      include_evaluator:bool=False)->dict:
        """PDFs and ONLY opaque IDs + questions in runner/. Labels in optional sibling.

        Never put the developer archive or evaluator/ in the service's input folder.
        Existing directories are rejected rather than potentially retaining old gold.
        """
        rows=self.questions(view,allow_audit=allow_audit)
        integrity=self.corpus.verify()
        if not integrity['ok']:raise ValueError('Corpus integrity failed before export: '+str(integrity['errors']))
        target=Path(output).expanduser().resolve()
        if target.exists():raise FileExistsError(f'Output already exists: {target}')
        target.parent.mkdir(parents=True,exist_ok=True)
        fingerprint=self.fingerprint()
        def opaque(prefix:str, value:str)->str:
            return prefix+hashlib.sha256((fingerprint+'\0'+value).encode()).hexdigest()[:20]
        doc_ids=sorted({q['document_id'] for q in rows})
        mapping={i:opaque('d_',i) for i in doc_ids}
        reverse=[]; runner=[]; pdfs=[]
        temporary=Path(tempfile.mkdtemp(prefix='.corpus-export-',dir=target.parent))
        try:
            for i in doc_ids:
                d=self.corpus.get(i);new=mapping[i]
                p=temporary/'runner/documents'/f'{new}.pdf';p.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(d.pdf_path,p)
                pdfs.append({'id':new,'file':f'documents/{new}.pdf','sha256':sha256(p)})
            for q in rows:
                new_id=opaque('q_',q['id'])
                runner.append({'id':new_id,'document_id':mapping[q['document_id']],'question':q['question']})
                reverse.append({'runner_question_id':new_id,'runner_document_id':mapping[q['document_id']],
                                'source_question_id':q['id'],'source_document_id':q['document_id'],
                                'family_id':q['family_id'],'suite':q['suite'],'split':q['split'],'gold':q['gold']})
            write_jsonl(temporary/'runner/questions.jsonl',runner)
            write_json(temporary/'runner/documents.json',pdfs)
            write_json(temporary/'runner/run_manifest.json',{'schema_version':'1.0','dataset_hash':fingerprint,
              'gold_version':self.corpus.manifest.get('gold_version'),'document_count':len(pdfs),'question_count':len(rows),
              'contains_gold':False,'question_contract':['id','document_id','question']})
            if include_evaluator:
                write_jsonl(temporary/'evaluator/private_mapping_and_gold.jsonl',reverse)
                write_json(temporary/'evaluator/protocol.json',{'view':view,'dataset_hash':fingerprint,
                    'aggregation':'repetitions -> question -> root family; equal family weights',
                    'do_not_upload':'This entire evaluator directory must stay outside the RAG ingestion inputs.',
                    'audit_is_not_secret':'Developer archive already contains all gold.'})
            temporary.rename(target)
        except BaseException:
            shutil.rmtree(temporary,ignore_errors=True);raise
        return {'output':str(target),'view':view,'documents':len(pdfs),'questions':len(rows),
                'dataset_hash':fingerprint,'runner':str(target/'runner'),'evaluator_included':include_evaluator}

    def review_queue(self, view:str='dev', *, allow_audit:bool=False)->list[dict]:
        """Human or independent-agent work items; a queue is NOT completed review."""
        queue=[]
        for q in self.questions(view,allow_audit=allow_audit):
            d=self.corpus.get(q['document_id'])
            digest=hashlib.sha256(json.dumps({'question':q['question'],'gold':q['gold']},sort_keys=True,ensure_ascii=False).encode()).hexdigest()
            queue.append({'id':q['id'],'document_id':d.id,'pdf_sha256':sha256(d.pdf_path),
                'question_and_gold_sha256':digest,'question':q['question'],'current_gold':q['gold'],
                'reviewer':'','reviewer_kind':None,'independent_of_author':False,
                'decision':None,'whole_document_checked':False,'alternative_evidence':[],
                'notes':'','review_status':'pending',
                'checks':['question unambiguous','all claims supported','qualifications preserved',
                          'negative checked across whole document','alternative support considered','physical pages visually checked']})
        return queue
