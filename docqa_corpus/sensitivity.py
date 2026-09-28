"""Matched question variations and controlled context probes, not new IID cases."""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
import random
from .benchmark import Benchmark
from .util import read_json, read_jsonl, write_jsonl, write_json


class SensitivitySuite:
    def __init__(self, corpus=None):
        self.benchmark=Benchmark(corpus)
        self.corpus=self.benchmark.corpus

    def query_variants(self, view:str='dev', *, allow_audit:bool=False)->list[dict]:
        selected={q['id'] for q in self.benchmark.questions(view,allow_audit=allow_audit)}
        path=self.corpus.root/'sensitivity/query_variants.jsonl'
        return [v for v in read_jsonl(path) if v['parent_question_id'] in selected]

    def contrast_groups(self)->list[dict]:
        value=read_json(self.corpus.root/'sensitivity/contrast_groups.json')
        return value['groups'] if isinstance(value,dict) else value

    def context_probe(self, question_id:str, *, remove_evidence_ids:list[str]|None=None,
                      order:str='source', seed:int=0, allow_audit:bool=False)->dict:
        """Context-level coverage. Removing evidence never changes document answerability.

        Candidate alternate unannotated support and entailment still need review.
        This produces evaluator fixtures, not the main service's answer contract.
        """
        rows={q['id']:q for q in self.corpus.questions()}
        if question_id not in rows:raise KeyError(question_id)
        q=rows[question_id]
        if q['split']=='holdout' and not allow_audit:raise PermissionError('Explicit audit opt-in required')
        g=q['gold']; ev={e['id']:e for e in g['evidence']}
        remove=set(remove_evidence_ids or [])
        if not remove<=set(ev):raise ValueError('Unknown evidence IDs in removal list')
        candidates=[e for key,e in ev.items() if key not in remove]
        if order=='source':candidates.sort(key=lambda e:e['char_start'])
        elif order=='reverse':candidates.sort(key=lambda e:e['char_start'],reverse=True)
        elif order=='shuffle':random.Random(seed).shuffle(candidates)
        else:raise ValueError('order must be source, reverse or shuffle')
        left={e['id'] for e in candidates}
        complete=[group for group in g['evidence_sets'] if set(group)<=left]
        return {'question_id':question_id,'question':q['question'],'context':candidates,
                'document_answerable':g['found'],'annotated_sufficient_set_present':bool(complete),
                'complete_evidence_sets':complete,'removed_evidence_ids':sorted(remove),
                'note':'Coverage of annotated evidence only; not a proof of intrinsic insufficiency or absence in the full document.'}


def build_query_variants(corpus)->dict:
    from .core import Corpus
    corpus=corpus if isinstance(corpus,Corpus) else Corpus(corpus)
    rows=[]
    for q in corpus.questions(suite='S'):
        for i,text in enumerate(q.get('paraphrases',[]),1):
            if not text.strip() or text==q['question']:raise ValueError(f'{q["id"]}: empty/unchanged paraphrase')
            rows.append({'id':q['id']+f'-P{i}','parent_question_id':q['id'],
              'document_id':q['document_id'],'family_id':q['family_id'],'split':q['split'],
              'question':text,'transformation':'authored_paraphrase','expected_relation':'same_facts_and_answerability',
              'gold_reference':q['id'],'gold_reference_scope':'answer facts and answerability; retrieval support must be reviewed for this wording',
              'independent_human_review':False,
              'count_as_independent_case':False})
        if q.get('paraphrases'):
            text=q['question'].replace(' ','\u00a0')
            if text!=q['question']:
                rows.append({'id':q['id']+'-W1','parent_question_id':q['id'],
                  'document_id':q['document_id'],'family_id':q['family_id'],'split':q['split'],
                  'question':text,'transformation':'nonbreaking_spaces','expected_relation':'same_facts_and_answerability',
                  'gold_reference':q['id'],'gold_reference_scope':'answer facts and answerability; retrieval support must be reviewed for this wording',
              'independent_human_review':False,'count_as_independent_case':False})
    write_jsonl(corpus.root/'sensitivity/query_variants.jsonl',rows)
    write_json(corpus.root/'sensitivity/protocol.json',{
      'query_variants':'Report paired semantic consistency and absolute correctness; do not reward consistently wrong answers.',
      'contrast':'Count a triplet successful only when all three states are semantically correct, including explicit negatives.',
      'M':'Changed facts must change answers; unchanged control facts must stay unchanged. Respect parent families.',
      'context':'Do not relabel full-document gold when masking retrieved context.',
      'collection_mix':'Freeze target PDF/index/model/config; add other documents; re-run target with same doc_id filter. Report changed ranking separately from changed answer.',
      'counts_are_clustered':True})
    return {'variants':len(rows),'parent_questions':len({r['parent_question_id'] for r in rows}),
            'independent_new_families':0}
