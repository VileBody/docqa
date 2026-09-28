"""Append-only release of unused closed-Pod reservations, never a bank refund."""
import time
from ..util import read,write,digest

def release_closed(budget,sessions):
    released=[]
    with budget._lock():
        ledger=read(budget.path)
        done={c.get('pod_id') for c in ledger['calls'] if c['kind']=='closed_pod_reservation_adjustment'}
        for state in sessions:
            pod=state.get('pod_id')
            if not pod or pod in done or state.get('status')!='deleted':continue
            if state.get('network_volumes'):continue
            ceiling=state['hourly_ceiling']
            # 40 GB container disk costs < $0.01/hour; retain original reservation
            # if quoted compute plus storage cannot fit its declared ceiling.
            if state.get('reported_hourly_usd',0)<=0 or state['reported_hourly_usd']+.01>ceiling:continue
            start=state.get('create_started_at',state.get('created_at'))
            if start is None or not state.get('deleted_at'):continue
            matches=[(i,c) for i,c in enumerate(ledger['calls']) if c['kind']=='runpod_session_upper_bound'
                     and 0<=start-c['started_at']<30 and c['reserved_usd']>0]
            if not matches:continue
            i,original=min(matches,key=lambda pair:start-pair[1]['started_at'])
            elapsed=max(0,state['deleted_at']-original['started_at'])+120
            bound=min(original['reserved_usd'],elapsed*ceiling/3600)
            adjustment=bound-original['reserved_usd']
            entry={'kind':'closed_pod_reservation_adjustment','pod_id':pod,'original_call_index':i,
                   'reserved_usd':adjustment,'closed_elapsed_ceiling_usd':bound,'status':'confirmed_deleted_upper_bound',
                   'started_at':time.time(),'finished_at':time.time(),'session_receipt_sha256':digest(state),
                   'method':'Original hourly ceiling times full create-to-confirmed-delete duration plus 120 seconds; original reservation retained in history. Compute and storage billed per second; no persistent storage.',
                   'billing_source':'https://github.com/runpod/docs/blob/main/accounts-billing/billing.mdx',
                   'invoice':False,'bank_refund':False}
            ledger['calls'].append(entry);ledger['reserved_usd']+=adjustment;done.add(pod);released.append(entry)
        assert ledger['reserved_usd']>=-1e-9
        write(budget.path,ledger);budget.state=ledger
    return released
