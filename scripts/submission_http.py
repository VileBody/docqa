"""Actual HTTP client. No rental. Live mode requires explicit scope/hash authorization."""
import argparse,json,os,time,hashlib
from pathlib import Path
import httpx
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def run(base,inputs,output,authorization=None):
    plan=json.loads(inputs.read_text());output.mkdir(parents=True,exist_ok=False)
    events=[]
    def record(**event):
        events.append(event);(output/'HTTP_OUTPUTS.json').write_text(json.dumps(events,ensure_ascii=False,indent=2)+'\n')
    with httpx.Client(base_url=base,headers={'Authorization':'Bearer '+os.environ['DOCQA_API_KEY']},timeout=240) as client:
        health=client.get('/health');health.raise_for_status();record(operation='health',response=health.json())
        if not health.json()['test_mode']:
            if authorization is None:raise ValueError('Live HTTP needs explicit authorization file')
            auth=json.loads(authorization.read_text())
            limit={'final_luna_p1_http_smoke_v1':5,'stronger_reader_v1':25}.get(auth.get('scope'),0)
            if not(auth.get('approved') is True and limit>0 and auth.get('inputs_sha256')==sha(inputs) and 0<auth.get('max_usd',0)<=limit):raise ValueError('Wrong live scope/inputs/budget')
            if auth['scope']=='stronger_reader_v1' and len(plan['items'])!=12:raise ValueError('Expected frozen 12-question series')
        documents={}
        for row in plan['items']:
            name=row['document_path']
            if name in documents:continue
            path=(ROOT/name).resolve()
            if not path.is_relative_to(ROOT.resolve()) or sha(path)!=row['document_sha256']:raise ValueError('Frozen document mismatch')
            start=time.monotonic()
            res=client.post('/documents',files={'file':(path.name,path.read_bytes(),'text/plain')},data={'classification':'public' if name.startswith('datasets/') else 'synthetic'})
            record(operation='upload',document=name,status=res.status_code,elapsed_s=time.monotonic()-start,response=res.json());res.raise_for_status()
            accepted=res.json();documents[name]=accepted['document']['doc_id'];deadline=time.monotonic()+900;states=[]
            while time.monotonic()<deadline:
                res=client.get('/tasks/'+accepted['task_id']);res.raise_for_status();value=res.json()
                if not states or states[-1]!=value['status']:states.append(value['status'])
                if value['status'] in {'ready','failed'}:break
                time.sleep(.5)
            record(operation='poll',document=name,states=states,response=value)
            if value['status']!='ready':raise RuntimeError('Indexing not ready')
        res=client.get('/documents');res.raise_for_status();record(operation='catalog',response=res.json())
        for row in plan['items']:
            # Evaluation fields, even if present in input, are never passed to API.
            body={'document_id':documents[row['document_path']],'question':row['question'],'diagnostics':True}
            start=time.monotonic()
            # H12 uses a fresh client; no implicit conversation/session identifiers.
            if row['id']=='H12':
                with httpx.Client(base_url=base,headers={'Authorization':'Bearer '+os.environ['DOCQA_API_KEY']},timeout=240) as fresh:res=fresh.post('/questions',json=body)
            else:res=client.post('/questions',json=body)
            record(operation='question',case_id=row['id'],request=body,status=res.status_code,elapsed_s=time.monotonic()-start,response=res.json())
            if res.status_code==200:
                text=(ROOT/row['document_path']).read_bytes().decode('utf8')
                for c in res.json()['citations']:
                    assert c['document_id']==body['document_id'] and text[c['char_start']:c['char_end']]==c['text']
                    assert text[:c['char_start']].count('\f')+1==c['page']
        receipt={'evidence_type':'real_infrastructure_model_doubles' if health.json()['test_mode'] else 'live_models_http','questions':len(plan['items']),'documents':len(documents),'statuses':[x['status'] for x in events if x['operation']=='question'],'semantic_review':'NOT_RUN' if health.json()['test_mode'] else 'PENDING_AGENT_ADJUDICATION','inputs_sha256':sha(inputs)}
        (output/'RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n');return receipt
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base-url',default='http://127.0.0.1:18080');p.add_argument('--inputs',type=Path,default=ROOT/'examples/submission/HTTP_INPUTS.json');p.add_argument('--output',type=Path,required=True);p.add_argument('--authorization',type=Path);a=p.parse_args()
    print(json.dumps(run(a.base_url,a.inputs,a.output,a.authorization),indent=2))
