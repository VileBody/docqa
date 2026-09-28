"""Offline corpus access. Loading and validating annotations needs only stdlib."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator
from .util import read_json, read_jsonl, write_jsonl, sha256, text_hash, safe_id


@dataclass(frozen=True)
class Document:
    root: Path
    metadata: dict[str, Any]

    @property
    def id(self)->str:return self.metadata['id']
    @property
    def pdf_path(self)->Path:return self.root/self.metadata['pdf']
    @property
    def available(self)->bool:return self.pdf_path.is_file()
    @property
    def source_dir(self)->Path:return self.root/self.metadata['source_dir']
    @property
    def canonical_text(self)->str:
        return (self.source_dir/'canonical.txt').read_text(encoding='utf-8')
    def questions(self, *, ready_only:bool=True)->list[dict]:
        path=self.root/self.metadata['annotations']
        rows=read_jsonl(path) if path.exists() else []
        return [r for r in rows if not ready_only or r['status'].startswith('ready')]


class Corpus:
    """Corpus directory or project directory; no implicit network access.

    Core imports are deliberately dependency-free. PDFs already in the archive
    can be uploaded straight to any API without installing the rendering extras.
    """
    def __init__(self,root:str|Path|None=None):
        p=Path(root).expanduser().resolve() if root else Path(__file__).resolve().parents[1]/'corpus'
        self.root=p/'corpus' if (p/'corpus/manifest.json').exists() else p
        if not (self.root/'manifest.json').is_file():
            raise FileNotFoundError(f'No manifest.json at {self.root}. Pass Corpus("/path/to/kit/corpus").')

    @property
    def manifest(self)->dict:return read_json(self.root/'manifest.json')

    def documents(self,*,suite:str|None=None,split:str|None=None,
                  available_only:bool=True)->Iterator[Document]:
        for item in self.manifest['documents']:
            if suite and item['suite']!=suite:continue
            if split and item['split']!=split:continue
            doc=Document(self.root,item)
            if not available_only or doc.available:yield doc

    def get(self,document_id:str)->Document:
        safe_id(document_id)
        for d in self.documents(available_only=False):
            if d.id==document_id:return d
        raise KeyError(document_id)

    def questions(self,*,document_id:str|None=None,suite:str|None=None,split:str|None=None,
                  include_mutations:bool=False,include_controls:bool=False,ready_only:bool=True)->Iterator[dict]:
        for doc in self.documents(suite=suite,split=split,available_only=ready_only):
            if document_id and doc.id!=document_id:continue
            if not include_mutations and suite!='M' and doc.metadata['suite']=='M':continue
            if not include_controls and suite!='C' and doc.metadata['suite']=='C':continue
            yield from doc.questions(ready_only=ready_only)

    def export_questions(self,path:str|Path,*,with_gold:bool=False,**filters:Any)->int:
        rows=list(self.questions(**filters))
        if not with_gold:
            rows=[{k:q[k] for k in ('id','document_id','family_id','split','question')} for q in rows]
        write_jsonl(Path(path),rows)
        return len(rows)

    def verify(self,*,require_public:bool=False)->dict:
        errors=[];warnings=[];n_docs=0;n_questions=0;n_evidence=0
        ids=set()
        for d in self.documents(available_only=False):
            if d.id in ids:errors.append(f'duplicate ID {d.id}')
            ids.add(d.id)
            if not d.available:
                msg=f'{d.id}: original PDF not present ({d.metadata["status"]})'
                (errors if require_public or d.metadata["suite"] != "R" else warnings).append(msg)
                continue
            n_docs+=1
            expected=d.metadata.get('sha256')
            if expected and sha256(d.pdf_path)!=expected:errors.append(f'{d.id}: PDF SHA-256 mismatch')
            if not (d.source_dir/'canonical.txt').is_file():
                warnings.append(f'{d.id}: no canonical text');continue
            text=d.canonical_text;h=text_hash(text)
            if d.metadata['suite']=='R' and len(d.questions()) < d.metadata['questions']:
                msg=f'{d.id}: official source questions are not all approved for scoring'
                (errors if require_public else warnings).append(msg)
            if d.metadata.get('canonical_sha256') and h!=d.metadata['canonical_sha256']:
                errors.append(f'{d.id}: canonical SHA-256 mismatch')
            seen=set()
            for row in d.questions():
                n_questions+=1
                if row['id'] in seen:errors.append(f'duplicate question ID {row["id"]}')
                seen.add(row['id'])
                g=row['gold']
                if g['found']!=bool(g['evidence']):errors.append(f'{row["id"]}: invalid found/evidence contract')
                ev_ids={e['id'] for e in g['evidence']}
                if g['found'] and not g['evidence_sets']:errors.append(f'{row["id"]}: missing evidence sets')
                for group in g['evidence_sets']:
                    if not set(group)<=ev_ids:errors.append(f'{row["id"]}: unresolved evidence set')
                for e in g['evidence']:
                    n_evidence+=1;a=e['char_start'];b=e['char_end']
                    if not 0<=a<b<=len(text) or text[a:b]!=e['text']:
                        errors.append(f'{row["id"]}: quote/offset mismatch: {e["id"]}')
                    if e['document_id']!=d.id or e['canonical_sha256']!=h:
                        errors.append(f'{row["id"]}: cross-document/stale evidence')
                    if not 1<=e['page']<=d.metadata['pages']:
                        errors.append(f'{row["id"]}: page out of range')
        return {'ok':not errors,'documents_checked':n_docs,'questions_checked':n_questions,
                'evidence_checked':n_evidence,'errors':errors,'warnings':warnings,
                'note':'Integrity checks are not a semantic-quality or independent human-review certificate.'}
