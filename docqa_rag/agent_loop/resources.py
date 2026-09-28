"""Placement/read-only catalog and zero-allocation accounting reused from stage2/3.
No new infrastructure backend or automatic non-idempotent create retry.
"""

import os, time, httpx

from ..util import read, write, digest

def placement_candidates():
 # Read-only native catalog; never infer capacity from a historical successful Pod.
 with httpx.Client(headers={'Authorization':'Bearer '+os.environ['RUNPOD_API_KEY']},timeout=30) as c:
  r=c.get('https://api.runpod.io/v2/catalog/gpus',params={'include':'AVAILABILITY','product':'POD','cloud':'SECURE','count':1,'cudaVersions':'12.8,12.9,13.0'});r.raise_for_status();catalog=r.json()
 allow=['NVIDIA RTX PRO 4500 Blackwell','NVIDIA RTX PRO 4500 Blackwell Server Edition','NVIDIA A40','NVIDIA RTX A6000','NVIDIA GeForce RTX 4090','NVIDIA GeForce RTX 3090','NVIDIA RTX 6000 Ada Generation','NVIDIA L40','NVIDIA L40S']
 found=[]
 for g in catalog['gpus']:
  if g['id'] in allow and g.get('secure') and g.get('memory',0)>=24 and 0<g.get('price',{}).get('secure',0)<=1.2:
   for d in g.get('dataCenters',[]):
    if d.get('availability') in {'LOW','MEDIUM','HIGH'}:found.append({'gpu':g['id'],'dc':d['id'],'quote':g['price']['secure']})
 found.sort(key=lambda x:(allow.index(x['gpu']),x['dc']));return catalog,found[:3]

def release_zero_allocation(budget,session,fresh_pods):
    """An append-only documented adjustment, only after definitive no-allocation."""
    s=session.state
    if s.get('pod_id') or s.get('create_failure',{}).get('http_status')!=400 or s.get('create_reconciliation',{}).get('matching_pod_ids')!=[] or any(p.get('name')==s.get('name') for p in fresh_pods):
        raise ValueError('Allocation ambiguous; never credit or create again')
    with budget._lock():
        state=read(budget.path)
        entry=state['calls'][-1]
        if entry['kind']!='runpod_session_upper_bound' or entry['status']!='reserved' or entry['reserved_usd']!=s['reserved_usd']:
            raise ValueError('Reservation mismatch')
        entry['status']='confirmed_no_allocation';entry['released_usd']=entry['reserved_usd']
        state['reserved_usd']-=entry['reserved_usd']
        state.setdefault('adjustments',[]).append({'reason':'HTTP400 and reconciled no matching pod; fresh inventory checked','released_usd':entry['reserved_usd'],
            'session_sha256':digest(session.path.read_bytes()),'observed_pod_ids':[p['id'] for p in fresh_pods],'at':time.time()})
        write(budget.path,state);budget.state=state
