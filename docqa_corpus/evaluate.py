"""Deterministic diagnostics, NOT an automated claim-entailment judge.

Exact citation existence, answerability and span coverage cannot certify that
an answer follows from its citations. semantic_correct is intentionally None.
"""
from __future__ import annotations
from collections import Counter
from pathlib import Path
from typing import Any
from .core import Corpus
from .util import read_json


def _coverage(start: int, end: int, intervals: list[tuple[int,int]]) -> float:
    clipped=sorted((max(start,a),min(end,b)) for a,b in intervals if a<end and b>start)
    total=0;right=start
    for a,b in clipped:
        left=max(a,right)
        if b>left:total+=b-left
        right=max(right,b)
    return total/(end-start) if end>start else 0.0


class AnswerEvaluator:
    def __init__(self,corpus: Corpus | str | Path | None = None):
        self.corpus=corpus if isinstance(corpus,Corpus) else Corpus(corpus)

    def check(self,question: dict, prediction: dict, *, context_spans: list[dict] | None = None) -> dict:
        if not question.get('status','').startswith('ready') or question.get('gold') is None:
            raise ValueError('Only ready, annotated questions may be scored')
        doc=self.corpus.get(question['document_id']);text=doc.canonical_text
        pages={p['page']:p for p in read_json(doc.source_dir/'pages.json')}
        gold=question['gold'];errors=[]
        found=prediction.get('found');answer=prediction.get('answer');citations=prediction.get('citations')
        if type(found) is not bool:errors.append('found_must_be_boolean')
        if not isinstance(answer,str) or not answer.strip():errors.append('answer_must_be_nonempty_string')
        if not isinstance(citations,list):errors.append('citations_must_be_list');citations=[]
        if found is True and not citations:errors.append('found_without_citations')
        if found is False and citations:errors.append('abstention_with_citations')
        if prediction.get('document_id',doc.id)!=doc.id:errors.append('wrong_document')
        details=[]
        for i,c in enumerate(citations):
            if not isinstance(c,dict):details.append({'index':i,'exists':False,'reason':'not_object'});continue
            page=c.get('page');quote=c.get('text');p=pages.get(page) if type(page) is int else None
            correct_doc=c.get('document_id',doc.id)==doc.id
            valid=bool(correct_doc and p and isinstance(quote,str) and quote and quote in p['text'])
            details.append({'index':i,'exists':valid,'reason':None if valid else 'wrong_document_page_or_quote'})
        if any(not c['exists'] for c in details):errors.append('invalid_citation')
        forbidden=[]
        if isinstance(answer,str):
            forbidden=[s for s in gold.get('forbidden_answer_substrings',[]) if s.casefold() in answer.casefold()]
        # Search-context coverage requires offsets in THIS corpus coordinate
        # system and a matching frozen canonical hash, not arbitrary backend IDs.
        retrieval=None
        if context_spans is not None:
            intervals=[];rejected=[]
            for i,s in enumerate(context_spans):
                a=s.get('char_start');b=s.get('char_end')
                valid=(s.get('document_id')==doc.id and s.get('canonical_sha256')==doc.metadata['canonical_sha256']
                       and type(a) is int and type(b) is int and 0<=a<b<=len(text))
                if valid:intervals.append((a,b))
                else:rejected.append(i)
            coverage={e['id']:_coverage(e['char_start'],e['char_end'],intervals) for e in gold['evidence']}
            complete=(any(all(coverage[eid]==1.0 for eid in group) for group in gold['evidence_sets'])
                      if gold['found'] else None)
            retrieval={'anchor_span_coverage':coverage,'all_required_anchor_spans_covered':complete,'rejected_context_indices':rejected}
        return {'question_id':question['id'],'document_id':doc.id,'schema_and_citations_valid':not errors,
                'errors':errors,'found_matches_gold':found==gold['found'] if type(found) is bool else False,
                'citation_checks':details,'forbidden_markers':forbidden,'retrieval':retrieval,
                'semantic_correct':None,
                'needs_semantic_review':'Verify every claim, scope, units, roles, exceptions and completeness; citation existence alone is insufficient.'}

    @staticmethod
    def summarize(results: list[dict]) -> dict:
        n=len(results);cites=[c for r in results for c in r['citation_checks']]
        return {'n':n,'schema_and_citation_valid_rate':sum(r['schema_and_citations_valid'] for r in results)/n if n else None,
                'answerability_match_rate':sum(r['found_matches_gold'] for r in results)/n if n else None,
                'exact_citation_existence_rate':sum(c['exists'] for c in cites)/len(cites) if cites else None,
                'errors':dict(Counter(e for r in results for e in r['errors'])),
                'semantic_accuracy':None,'note':'These are deterministic diagnostics, not factual correctness or legal reliability.'}
