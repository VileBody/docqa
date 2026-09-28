"""Reproducible metadata repair and evaluation views. No semantic certification."""
from __future__ import annotations
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from .core import Corpus
from .util import read_json, read_jsonl, write_json, write_jsonl

VERSION = '0.2.0'
CRITICAL_SLICES = ('table', 'exception', 'version', 'long', 'conflict',
                   'multi_evidence', 'attribution', 'page_boundary')
OLD_GENRES = {'S01':'contract','S02':'procedure','S03':'accounting_policy',
    'S04':'financial_table','S05':'procedure','S06':'procedure',
    'S07':'procedure','S08':'contract','S09':'procedure',
    'S10':'procedure','S11':'contract','S12':'long_mixed'}


def evidence_features(row: dict) -> dict[str, Any]:
    gold = row.get('gold') or {}
    evidence = {e['id']:e for e in gold.get('evidence', [])}
    groups = gold.get('evidence_sets', [])
    if any(not g or not set(g) <= set(evidence) for g in groups):
        raise ValueError(f'{row["id"]}: invalid/empty evidence alternative')
    if gold.get('found') and not groups:
        raise ValueError(f'{row["id"]}: answerable question without evidence sets')
    return {
        'evidence_spans_listed':len(evidence),
        'alternative_sufficient_sets':len(groups),
        'min_required_spans': min(map(len, groups), default=0),
        'min_required_physical_pages': min((len({evidence[k]['page'] for k in g}) for g in groups), default=0),
        'found':gold.get('found'),
        'mode':gold.get('mode'),
        'derivation':'v2 AND within a set; OR between sets; not semantic entailment testing',
    }


def curate(corpus: Corpus | str | Path) -> dict:
    corpus = corpus if isinstance(corpus, Corpus) else Corpus(corpus)
    manifest = corpus.manifest
    changes=[]
    for d in manifest['documents']:
        if d['suite']=='R':
            d.update(benchmark_role='quarantined_public', author_batch_id='external_original_not_yet_acquired')
            continue
        source_path=corpus.root/d['source_dir']/'source.json'
        source=read_json(source_path)
        d['genre']=source.get('genre', OLD_GENRES.get(d['family_id'],'contrast_control'))
        d['author_batch_id']=source.get('author_batch_id','assistant_authored_v1')
        d['template_group_id']=source.get('template_group_id',d['family_id']+'-scenario')
        d['independent_source_author']=False
        if d['suite']=='M':role='metamorphic_regression'
        elif d['suite']=='C':role='contrast_control'
        elif int(d['id'][1:])<=12 and d['split']=='holdout':role='legacy_exposed_audit'
        elif d['split']=='holdout':role='frozen_authored_audit'
        else:role='development'
        d['benchmark_role']=role
        p=corpus.root/d['annotations']; rows=read_jsonl(p)
        for q in rows:
            original=set(q['tags']); features=evidence_features(q)
            tags=set(original)
            if features['min_required_spans']>1:tags.add('multi_evidence')
            else:tags.discard('multi_evidence')
            if features['alternative_sufficient_sets']>1:tags.add('alternative_evidence')
            if features['min_required_physical_pages']>1:tags.add('multi_page_evidence')
            if features['mode']=='conflict':tags.add('conflict')
            q['tags']=sorted(tags)
            q['features']=features
            q['gold_version']='authored-v2.0'
            q['benchmark_role']=role
            q['genre']=d['genre']
            q['author_batch_id']=d['author_batch_id']
            q['review'].setdefault('independent_human_review', False)
            if q['review']['independent_human_review'] is not True:
                q['review']['reviewer_kind']='authoring_agent'
                q['review']['method']='Authored specification; mechanical quote/coordinate checks. Not independent semantic review.'
                q['review']['semantic_review_status']='author_checked_not_independently_adjudicated'
            if not q['gold']['found']:
                q.setdefault('negative_reason','Legacy authored absence; requires independent whole-document confirmation.')
            if original!=tags:
                changes.append({'id':q['id'],'added':sorted(tags-original),'removed':sorted(original-tags),
                                'reason':'Derived evidence-set structure, not number of listed citations'})
        write_jsonl(p, rows)
    manifest.update(version=VERSION, gold_version='authored-v2.0', schema_version='1.1',
        annotation_semantics='evidence_sets are OR alternatives of AND-required evidence IDs',
        release_scope='Synthetic diagnostic expansion; no external/human validation claim')
    write_json(corpus.root/'manifest.json',manifest)
    byid={d['id']:d for d in manifest['documents']}
    dev=[d['id'] for d in manifest['documents'] if d.get('benchmark_role')=='development']
    audit=[d['id'] for d in manifest['documents'] if d.get('benchmark_role')=='frozen_authored_audit']
    legacy=[d['id'] for d in manifest['documents'] if d.get('benchmark_role')=='legacy_exposed_audit']
    views={
        'smoke':{'documents':['S01','S08','S10','S18','S29','S33'], 'purpose':'Small diagnostic subset; not a statistically representative sample'},
        'dev':{'documents':dev,'purpose':'Development only; family-macro aggregation'},
        'audit':{'documents':audit,'requires_explicit_opt_in':True,'purpose':'New authored frozen diagnostic. Not independently authored, secret, or external'},
        'legacy':{'documents':legacy,'requires_explicit_opt_in':True,'purpose':'Previously discussed authored holdout; report separately'},
        'mutations':{'documents':[d['id'] for d in manifest['documents'] if d['suite']=='M'], 'purpose':'Paired regression, excluded from primary score'},
        'contrast':{'documents':[d['id'] for d in manifest['documents'] if d['suite']=='C'], 'purpose':'Same-question 3-state controls; score each triplet jointly'},
        'public':{'documents':[d['id'] for d in manifest['documents'] if d['suite']=='R'], 'purpose':'Real originals, blocked until acquired and independently reviewed'},
    }
    registry={'schema_version':'1.0','version':VERSION,'views':views,
        'independence_unit':'root scenario family; shared synthetic author still induces dependence',
        'primary_aggregation':'repeats -> question -> equally weighted root family; no pooling M/C',
        'audit_note':'All source and gold files are in the developer archive. Access guard is a workflow check, not security.'}
    write_json(corpus.root/'views.json', registry)
    write_json(corpus.root.parent/'reports/annotation_repairs.json',{'version':VERSION,'changes':changes})
    return {'documents':len(manifest['documents']), 'annotation_tag_changes':len(changes)}


def coverage(corpus: Corpus | str | Path) -> dict:
    corpus=corpus if isinstance(corpus,Corpus) else Corpus(corpus)
    views=read_json(corpus.root/'views.json')['views']; result={}
    for name,v in views.items():
        docs=[corpus.get(i) for i in v['documents']]
        rows=[q for d in docs if d.available for q in d.questions()]
        tags=defaultdict(set)
        for q in rows:
            for tag in q['tags']:tags[tag].add(q['family_id'])
        result[name]={'documents':len(docs),'available_documents':sum(d.available for d in docs),
          'questions':len(rows),'families':len({q['family_id'] for q in rows}),
          'answerable':sum(q['gold']['found'] for q in rows),
          'unanswerable':sum(not q['gold']['found'] for q in rows),
          'conflict_questions':sum(q['gold']['mode']=='conflict' for q in rows),
          'independent_human_reviewed_questions':sum(q['review'].get('independent_human_review') is True for q in rows),
          'genres':dict(Counter(d.metadata.get('genre','unclassified') for d in docs)),
          'critical_slice_families':{t:sorted(tags[t]) for t in CRITICAL_SLICES}}
    checks=[]
    for view in ('dev','audit'):
        for tag in CRITICAL_SLICES:
            n=len(result[view]['critical_slice_families'][tag]);checks.append({'view':view,'slice':tag,'families':n,'pass':n>=2})
    return {'views':result,'coverage_floor':'at least 2 authored families per critical slice per dev/audit; not a power calculation',
      'coverage_checks':checks,'coverage_floor_met':all(c['pass'] for c in checks),
      'external_validity_proven':False,'notes':[
        'Synthetic families are scenarios, not independently authored sources.',
        'M/C/query variants do not increase primary independent sample size.',
        'A future real-source acquisition quota is not fulfilled by new synthetic documents.',
        'No RAG systems have been run by this corpus preparation step.']}
