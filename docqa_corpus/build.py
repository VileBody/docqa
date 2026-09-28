"""Build complete synthetic bundles, including resolved evidence coordinates."""
from __future__ import annotations
from pathlib import Path
from typing import Any
from .util import read_json, write_json, write_jsonl, sha256, compact
from .render import render_document


def build_bundle(source: dict, qa_spec: list[dict], corpus_root: Path, *,
                 font_path: str | Path | None = None, bold_path: str | Path | None = None,
                 extra_provenance: dict | None = None, verify_text_layer: bool = True) -> dict[str,Any]:
    from pypdf import PdfReader
    id=source['id']; side=corpus_root/'sources'/id
    pdf=corpus_root/'documents'/f'{id}.pdf'
    meta=render_document(source,pdf,side,font_path,bold_path)
    if verify_text_layer:
        reader=PdfReader(pdf)
        if len(reader.pages)!=meta['pages']:
            raise ValueError(f'{id}: wrong page count')
        extracted=[compact(p.extract_text() or '') for p in reader.pages]
        missing=[a['block_id'] for a in meta['anchors'].values()
                 if compact(a['text']) not in extracted[a['page']-1]]
        if missing:
            raise ValueError(f'{id}: authored blocks absent from PDF text layer: {missing}')
    annotations=[]
    for spec in qa_spec:
        evidences=[]
        for key in spec['anchor_ids']:
            if key not in meta['anchors']:
                raise ValueError(f'{spec["id"]}: unknown anchor {key}')
            a=meta['anchors'][key]
            evidences.append({'id':f'{id}::{key}','document_id':id,
                'block_id':key,'text':a['text'],'page':a['page'],'printed_page':a['printed_page'],
                'section':a['section'],'char_start':a['char_start'],'char_end':a['char_end'],
                'bbox':a['bbox'],'bbox_origin':a['bbox_origin'],
                'canonical_sha256':meta['canonical_sha256']})
        found=spec['mode']!='abstain'
        if found != bool(evidences):
            raise ValueError(f'{spec["id"]}: found and evidence presence disagree')
        row={'id':spec['id'],'document_id':id,'family_id':source['family_id'],
             'suite':source.get('suite', 'M' if source.get('kind')=='mutation' else 'S'),'split':source['split'],
             'question':spec['question'],'tags':spec['tags'],'status':'ready_authored',
             'coordinate_system':'authored_source_v1',
             'review':{'method':'Authored facts plus mechanical PDF text-layer consistency; no independent semantic review',
                       'independent_human_review':False,
                       'negative_scope':'entire authored document' if not found else None},
             'gold':{'found':found,'mode':spec['mode'],'answer':spec['answer'],
                     'required_facts':spec['required_facts'],'evidence':evidences,
                     'evidence_sets':[[f'{id}::{key}' for key in group] for group in spec.get('evidence_options', [spec['anchor_ids']])] if found else [],
                     'forbidden_answer_substrings':spec.get('forbidden_answer_substrings',[])}}
        for key in ('parent_question_id','relation','question_rewrite_reason','paraphrases','negative_reason','negative_search_terms','intent_id','evaluation_role'):

            if key in spec:row[key]=spec[key]
        annotations.append(row)
    write_jsonl(corpus_root/'annotations'/f'{id}.jsonl',annotations)
    write_json(side/'qa_spec.json',qa_spec)
    write_json(side/'facts.json',source.get('facts',{}))
    prov={'document_id':id,'kind':source.get('kind','synthetic'),
          'created_on':'2026-09-24','generator_version':'0.2.0',
          'pdf_sha256':sha256(pdf),'canonical_sha256':meta['canonical_sha256'],
          'source_sha256':sha256(side/'source.json'),'physical_pages':meta['pages'],
          'characters':meta['characters'],'whitespace_words':meta['whitespace_words'],
          'token_count':None,'token_count_note':'Tokenizer-dependent; not estimated as an exact token count.',
          'font_used':meta['font_family_used'],'font_files_bundled':False,
          'text_layer_consistency_verified':verify_text_layer,
          'independent_human_review':False,'license':'MIT; fictional educational content',
          **(extra_provenance or {})}
    write_json(side/'provenance.json',prov)
    item={'id':id,'suite':source.get('suite', 'M' if source.get('kind')=='mutation' else 'S'),
          'title':source['title'],'family_id':source['family_id'],'split':source['split'],
          'status':'ready','pdf':f'documents/{id}.pdf','source_dir':f'sources/{id}',
          'annotations':f'annotations/{id}.jsonl','pages':meta['pages'],
          'questions':len(annotations),'sha256':prov['pdf_sha256'],
          'canonical_sha256':meta['canonical_sha256'],
          **{k:source[k] for k in ('genre','author_batch_id','template_group_id','benchmark_role') if k in source}}
    return item


def register(corpus_root:Path,item:dict)->None:
    path=corpus_root/'manifest.json'
    manifest=read_json(path) if path.exists() else {
        'schema_version':'1.0','version':'0.2.0','created_on':'2026-09-24',
        'coordinate_contract':{'page':'physical, 1-based','offsets':'Unicode codepoints, half-open',
                               'source':'authored source text for S/M, frozen extracted text for R after download',
                               'page_separator':'\\n\\f\\n'},
        'documents':[]}
    manifest['documents']=[x for x in manifest['documents'] if x['id']!=item['id']]+[item]
    manifest['documents'].sort(key=lambda x:x['id'])
    write_json(path,manifest)
