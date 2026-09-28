"""Tiny deterministic question-only diagnostic. Not a RAG or statistical proof."""
from __future__ import annotations
from collections import Counter
import math
import re
from typing import Any
from .benchmark import Benchmark


def question_only_probe(corpus=None)->dict[str,Any]:
    """Leave-one-root-family-out multinomial NB; DEV ONLY, no tuning/holdout."""
    rows=Benchmark(corpus).questions('dev')
    families=sorted({r['family_id'] for r in rows});predictions=[]
    if len(families)<3:raise ValueError('At least three development families needed')
    tokenize=lambda s:re.findall(r'[а-яёa-z]+',s.lower())
    for family in families:
        train=[q for q in rows if q['family_id']!=family]
        test=[q for q in rows if q['family_id']==family]
        ns=Counter();vocabulary=set();counts={0:Counter(),1:Counter()}
        for q in train:
            label=int(q['gold']['found']);tokens=tokenize(q['question'])
            ns[label]+=1;counts[label].update(tokens);vocabulary.update(tokens)
        denom={l:sum(counts[l].values())+len(vocabulary)+1 for l in (0,1)}
        for q in test:
            tokens=tokenize(q['question'])
            scores={l:math.log((ns[l]+1)/(len(train)+2))+
                       sum(math.log((counts[l][t]+1)/denom[l]) for t in tokens) for l in (0,1)}
            predictions.append({'id':q['id'],'family_id':family,'gold_found':int(q['gold']['found']),
                'predicted_found':max(scores,key=scores.get),'fold_majority_found':int(ns[1]>=ns[0])})
    def metrics(key):
        recalls={l:sum(r[key]==l for r in predictions if r['gold_found']==l)/sum(r['gold_found']==l for r in predictions) for l in (0,1)}
        return {'accuracy':sum(r[key]==r['gold_found'] for r in predictions)/len(predictions),
                'balanced_accuracy':sum(recalls.values())/2,'recall_unanswerable':recalls[0],'recall_answerable':recalls[1]}
    return {'method':'Fixed unigram multinomial Naive Bayes + Laplace(1), leave-one-family-out, dev only',
            'questions':len(rows),'families':len(families),'features':'question letters only; no PDF/answer/ID/source',
            'model':metrics('predicted_found'),'majority_baseline':metrics('fold_majority_found'),
            'predictions':predictions,'holdout_used':False,'hyperparameters_tuned':False,
            'interpretation':'A cheap proxy for lexical answerability bias. Above-chance balanced accuracy is a warning, not proof of leakage; failure to beat the majority accuracy is not proof of no bias.',
            'representativeness_certified':False,'rag_measured':False}
