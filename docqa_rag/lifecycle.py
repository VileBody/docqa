"""Bounded RunPod session. Owned IDs only, delete (never billable stop).

The worker runs its own deadline/idle/client-loss watchdog. No network volumes.
Failures of the provider API are recorded as blockers, not zero-cost guarantees.
"""
from __future__ import annotations
import base64,json,math,os,re,secrets,threading,time,uuid
from pathlib import Path
import httpx
from .util import read,write,digest,BudgetExceeded

def safe_error_body(response, secrets_to_redact):
    """Bounded diagnostic, not a dump of headers or an echoed create payload."""
    sensitive=re.compile(r'authorization|cookie|secret|token|api.?key|password|^env$|dockerStartCmd',re.I)
    def clean(value):
        if isinstance(value,dict):return {k:'[REDACTED]' if sensitive.search(str(k)) else clean(v) for k,v in value.items()}
        if isinstance(value,list):return [clean(v) for v in value]
        if isinstance(value,str):
            for secret in sorted(filter(None,secrets_to_redact),key=len,reverse=True):value=value.replace(secret,'[REDACTED]')
            value=re.sub(r'(?i)Bearer\s+[^\s"<>]+','Bearer [REDACTED]',value)
            return re.sub(r'(?i)((?:api[_-]?key|token|password|secret)\s*=\s*)[^\s&<>]+',r'\1[REDACTED]',value)
        return value
    try:body=clean(response.json());text=json.dumps(body,ensure_ascii=False)
    except (ValueError,TypeError):text=clean(response.text)
    text=clean(text)
    return {'body_sanitized':text[:4096],'body_truncated':len(text)>4096,'body_original_bytes':len(response.content)}

class RunPodRequestError(RuntimeError):
    def __init__(self,receipt):
        self.receipt=receipt
        super().__init__('RunPod HTTP '+str(receipt.get('http_status'))+': '+receipt.get('body_sanitized',receipt.get('error_type',''))[:500])

class RunPodAPI:
    base_url='https://rest.runpod.io/v1'
    def __init__(self,key): self.key=key
    def request(self,method,path,payload=None):
        # Do not retry non-idempotent create. Reconcile by unique session name instead.
        started=time.time()
        with httpx.Client(timeout=20,follow_redirects=False) as c:
            try:r=c.request(method,self.base_url+path,json=payload,headers={'Authorization':'Bearer '+self.key})
            except httpx.TransportError as exc:
                raise RunPodRequestError({'method':method,'path':path,'started_at':started,'finished_at':time.time(),
                    'http_status':None,'error_type':type(exc).__name__,'classification':'transport_outcome_unknown'}) from None
            if r.status_code==404 and method in {'GET','DELETE'}: return None
            if r.is_error:
                redactions=[self.key]+[str(v) for v in (payload or {}).get('env',{}).values()]
                body=safe_error_body(r,redactions)
                classification={400:'invalid_request_unresolved',401:'authentication',403:'authorization',402:'billing',429:'rate_limit'}.get(r.status_code,'server_or_other_error')
                if r.status_code>=500 and 'no instances currently available' in body['body_sanitized'].lower():classification='no_placement_for_request'
                raise RunPodRequestError({'method':method,'path':path,'started_at':started,'finished_at':time.time(),
                    'http_status':r.status_code,'classification':classification,**body})
            return r.json() if r.content else None

class Session:
    def __init__(self,path,api,budget,*,duration_s=1800,idle_s=240,hourly_ceiling=2,tail_loss_accepted=False,budget_derived_duration=False):
        self.path=Path(path); self.api=api; self.budget=budget; self.stop_event=threading.Event(); self.thread=None; self.token=None
        maximum=(budget.state['max_usd']-budget.state['reserved_usd'])*3600/hourly_ceiling if budget_derived_duration and hourly_ceiling>0 else 3600
        if not all(math.isfinite(v) for v in [duration_s,hourly_ceiling]) or not 60<=duration_s<=maximum or not 60<=idle_s<=600 or hourly_ceiling<=0: raise ValueError('Invalid lifecycle limits')
        self.state=read(path) if self.path.exists() else {'session_id':uuid.uuid4().hex,'pod_id':None,'owned':False,
            'duration_s':duration_s,'idle_s':idle_s,'hourly_ceiling':hourly_ceiling,'tail_loss_accepted':tail_loss_accepted,
            'deadline_epoch':None,'status':'planned','export_verified':False,'network_volumes':[],
            'emergency_policy':'delete after bounded retries; unsaved tail may be lost at hard deadline',
            'cleanup_attempts':[],'account_status':'not_checked'}
        write(self.path,self.state)
    def plan(self):
        return {**self.state,'max_compute_storage_usd':self.state['duration_s']/3600*self.state['hourly_ceiling'],
            'watchdog':'remote worker before install; deadline + idle + heartbeat loss','create_retries':0,'cleanup_retries':3}
    def up(self,*,image,gpu_type,bindings,worker_file,data_center_ids=None,cloud_type='SECURE',container_disk_gb=40):
        s=self.state
        if isinstance(container_disk_gb,bool) or not isinstance(container_disk_gb,int) or not 40<=container_disk_gb<=200:
            raise ValueError('Container disk must be an integer in 40..200 GB')
        if cloud_type not in {'SECURE','COMMUNITY'}:raise ValueError('Invalid cloud type')
        gpu_types=[gpu_type] if isinstance(gpu_type,str) else gpu_type
        if not isinstance(gpu_types,list) or not gpu_types or any(not isinstance(g,str) or not g for g in gpu_types):
            raise ValueError('Nonempty GPU type list required')
        if data_center_ids is not None and (not isinstance(data_center_ids,list) or not data_center_ids or any(not isinstance(d,str) or not d for d in data_center_ids)):
            raise ValueError('Nonempty data center list required')
        if s['pod_id']: raise ValueError('Session already has a Pod; use status/down')
        if s['status']!='planned': raise ValueError('Ambiguous previous create; reconcile manually before retry')
        if not s['tail_loss_accepted']: raise ValueError('Emergency tail-loss policy must be selected before rent')
        if '@sha256:' not in image: raise ValueError('Pin serving image by digest before rent')
        ceiling=s['duration_s']/3600*s['hourly_ceiling']
        self.budget.reserve('runpod_session_upper_bound',ceiling)
        deadline=min(time.time()+s['duration_s'],self.budget.state['deadline_epoch'])
        if deadline-time.time()<50: raise BudgetExceeded('Insufficient session time')
        self.token=secrets.token_urlsafe(32)
        s.update(status='creating',deadline_epoch=deadline,name='docqa-s1-'+s['session_id'],image=image,gpu_type=gpu_type,
                 worker_sha256=digest(Path(worker_file).read_bytes()),reserved_usd=ceiling)
        write(self.path,s)
        source=base64.b64encode(Path(worker_file).read_bytes()).decode()
        # The command contains only code; credentials go in provider env and are never written in manifest.
        command='import base64; exec(compile(base64.b64decode('+repr(source)+'),"worker.py","exec"))'
        payload={'name':s['name'],'imageName':image,'gpuTypeIds':gpu_types,'gpuCount':1,'cloudType':cloud_type,
            'allowedCudaVersions':['12.8','12.9','13.0'],'containerDiskInGb':container_disk_gb,'volumeInGb':0,'ports':['8000/http'],'dockerEntrypoint':['python','-u','-c'],
            'dockerStartCmd':[command],'env':{'RUNPOD_API_KEY':self.api.key,'DOCQA_DEADLINE':str(deadline),
                'DOCQA_IDLE_SECONDS':str(s['idle_s']),'DOCQA_WORKER_TOKEN':self.token,'DOCQA_SESSION_ID':s['session_id'],
                'DOCQA_MODEL_BINDINGS':json.dumps(bindings)}}
        if data_center_ids is not None:payload['dataCenterIds']=data_center_ids
        s['create_request']={k:v for k,v in payload.items() if k not in {'env','dockerStartCmd'}}
        s['create_request']['omitted_sensitive_fields']=['env','dockerStartCmd']
        s['control_api']=getattr(self.api,'base_url','test_double')
        if hasattr(self.api,'create_payload'):
            s['wire_create_request']={k:v for k,v in self.api.create_payload(payload).items() if k not in {'env','cmd','args'}}
        s['create_started_at']=time.time();write(self.path,s)
        try:
            pod=self.api.request('POST','/pods',payload)
        except BaseException as exc:
            s['create_failure']=getattr(exc,'receipt',{'error_type':type(exc).__name__,'diagnostic_body':'unavailable'})
            s['status']='create_ambiguous';write(self.path,s)
            # Creation may have succeeded while response was lost. Never issue a second POST.
            try:matches=[p for p in self.api.request('GET','/pods') or [] if p.get('name')==s['name']]
            except Exception as reconcile_error:
                s['create_reconciliation']={'status':'unavailable','error_type':type(reconcile_error).__name__}
                write(self.path,s);raise exc from None
            s['create_reconciliation']={'status':'checked','checked_at':time.time(),'matching_pod_ids':[p['id'] for p in matches]}
            if len(matches)!=1:
                s['status']='create_ambiguous'; write(self.path,s); raise
            pod=matches[0]
        s.update(pod_id=pod['id'],owned=True,status='created',created_at=time.time()); write(self.path,s)
        s['provider_hardware']={k:v for k,v in (pod.get('machine') or {}).items() if k in {'gpuTypeId','gpuDisplayName','cudaVersion','dataCenterId','memoryInGb','vcpuCount'}}
        rate=float(pod.get('costPerHr',0) or 0)
        if rate<=0 or rate>s['hourly_ceiling']:
            self.down(emergency=True); raise ValueError('Actual Pod rate missing/above approved ceiling')
        s['reported_hourly_usd']=rate; s['endpoint']='https://'+pod['id']+'-8000.proxy.runpod.net'; write(self.path,s)
        self._heartbeat_start()
        return dict(s)
    def _heartbeat_start(self):
        def loop():
            while not self.stop_event.is_set():
                if time.time()>=self.state['deadline_epoch']:
                    self.down(emergency=True); return
                try:
                    with httpx.Client(timeout=5) as c: c.post(self.state['endpoint']+'/heartbeat',headers={'Authorization':'Bearer '+self.token})
                except Exception: pass
                self.stop_event.wait(15)
        self.thread=threading.Thread(target=loop,daemon=True); self.thread.start()
    def readiness(self):
        while time.time()<self.state['deadline_epoch']-60:
            try:
                with httpx.Client(timeout=10) as c:
                    r=c.get(self.state['endpoint']+'/health',headers={'Authorization':'Bearer '+self.token}); data=r.json()
                if r.status_code==200 and data.get('ready') and data.get('session_id')==self.state['session_id']:
                    self.state.update(status='ready',health=data); write(self.path,self.state); return data
            except Exception: pass
            time.sleep(5)
        raise TimeoutError('Model readiness exceeded bounded session')
    def verify_local(self,files):
        if not files: raise ValueError('No local artifacts')
        checks={str(Path(f).resolve()):digest(Path(f).read_bytes()) for f in files}
        self.state.update(export_verified=True,local_artifacts=checks); write(self.path,self.state)
        return checks
    def down(self,emergency=False):
        s=self.state; self.stop_event.set()
        if not s.get('pod_id') or not s.get('owned'): return self.status()
        if not s['export_verified'] and not (emergency and s['tail_loss_accepted']): raise ValueError('Verify local export before normal deletion')
        for attempt in range(3):
            try:
                pod=self.api.request('GET','/pods/'+s['pod_id'])
                if pod and pod.get('name')!=s['name']: raise ValueError('Ownership mismatch; refusing delete')
                if pod: self.api.request('DELETE','/pods/'+s['pod_id'])
                if self.api.request('GET','/pods/'+s['pod_id']) is not None: raise RuntimeError('Deletion not yet confirmed')
                s['status']='deleted'; s['deleted_at']=time.time(); write(self.path,s); return self.status()
            except Exception as e:
                s['cleanup_attempts'].append({'attempt':attempt+1,'error_type':type(e).__name__}); write(self.path,s)
                time.sleep(min(2**attempt,4))
        s['status']='cleanup_blocked'; s['manual_command']='runpodctl pod delete '+s['pod_id']; write(self.path,s)
        return dict(s)
    def status(self):
        try:
            pods=self.api.request('GET','/pods') or []
            volumes=self.api.request('GET','/networkvolumes') or []
            endpoints=self.api.request('GET','/endpoints') or []
            self.state['account_status']={'pods':[p['id'] for p in pods],'networkvolumes':[p['id'] for p in volumes],'endpoints':[p['id'] for p in endpoints]}
        except Exception:
            self.state['account_status']='unavailable_no_zero_cost_claim'
        write(self.path,self.state); return dict(self.state)
