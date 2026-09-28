"""Opt-in SDK transport: short HTTPS submit/poll calls, unchanged completion JSON."""
import hashlib
import json
import math
import time
from urllib.parse import urlparse
import httpx


class ReaderTransportError(RuntimeError):
    pass


class ReaderJobTransport(httpx.BaseTransport):
    def __init__(self, base, key, *, session_id, worker_instance, request_namespace,
                 deadline_epoch, max_wait_s=900, poll_s=2, http_timeout_s=10,
                 client=None, clock=time.time, sleep=time.sleep):
        parsed=urlparse(base)
        if parsed.scheme!='https' and not (parsed.scheme=='http' and parsed.hostname in {'127.0.0.1','localhost','::1'}):
            raise ValueError('HTTPS required except local protected tunnel')
        if parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.path not in {'','/','/v1','/v1/'}:
            raise ValueError('Plain worker origin or /v1 endpoint required')
        if not all(isinstance(x,str) and x for x in [key,session_id,worker_instance,request_namespace]):
            raise ValueError('Explicit session, worker instance and durable case namespace required')
        if not all(math.isfinite(x) and x>0 for x in [deadline_epoch,max_wait_s,poll_s,http_timeout_s]):
            raise ValueError('Finite positive time bounds required')
        self.origin=base.rstrip('/').removesuffix('/v1')
        self.session_id,self.instance,self.namespace=session_id,worker_instance,request_namespace
        self.deadline,self.max_wait,self.poll_s=deadline_epoch,max_wait_s,poll_s
        self.timeout=min(http_timeout_s,20)
        self.clock,self.sleep=clock,sleep
        self.client=client or httpx.Client(follow_redirects=False)
        self.headers={'Authorization':'Bearer '+key,'X-DocQA-Worker-Instance':worker_instance}
        self.receipts=[]

    def close(self):
        self.client.close()

    def _request(self, method, path, end, body=None):
        remaining=end-self.clock()
        if remaining<=0:
            raise ReaderTransportError('job_wait_deadline_outcome_unknown')
        started=self.clock()
        try:
            response=self.client.request(method,self.origin+path,headers=self.headers,
                json=body,timeout=min(self.timeout,remaining),follow_redirects=False)
        except httpx.TransportError as exc:
            self.receipts.append({'method':method,'path':path,'error_type':type(exc).__name__,'at':started})
            raise
        self.receipts.append({'method':method,'path':path,'http_status':response.status_code,'at':started,
                              'elapsed_s':self.clock()-started})
        return response

    def handle_request(self, request):
        if request.method!='POST' or request.url.path!='/v1/chat/completions' or str(request.url).split('/v1/')[0]!=self.origin:
            raise ReaderTransportError('request_outside_reader_route')
        payload=json.loads(request.read())
        encoded=json.dumps(payload,sort_keys=True,ensure_ascii=False,allow_nan=False)
        job_id=hashlib.sha256(json.dumps([self.session_id,self.namespace,encoded],ensure_ascii=False).encode()).hexdigest()
        body={'session_id':self.session_id,'job_id':job_id,'request':payload}
        end=min(self.deadline-30,self.clock()+self.max_wait)
        row=None
        # All retries refer to the same durable job, never a fresh generation ID.
        for attempt in range(3):
            try:
                response=self._request('POST','/v1/reader/jobs',end,body)
                if response.status_code not in {200,202}:
                    if response.status_code in {502,503,504,524} and attempt<2:
                        self.sleep(min(self.poll_s,max(0,end-self.clock())));continue
                    raise ReaderTransportError('job_submit_http_'+str(response.status_code))
                row=response.json();break
            except httpx.TransportError:
                if attempt==2:
                    raise ReaderTransportError('job_submit_outcome_unknown') from None
                self.sleep(min(self.poll_s,max(0,end-self.clock())))
        if row is None:
            raise ReaderTransportError('job_submit_outcome_unknown')
        expected_hash=hashlib.sha256(encoded.encode()).hexdigest()
        while True:
            if row.get('session_id')!=self.session_id or row.get('job_id')!=job_id or row.get('request_sha256')!=expected_hash:
                raise ReaderTransportError('job_identity_mismatch')
            status=row.get('status')
            if status=='completed':
                result=row.get('result')
                if not isinstance(result,dict):raise ReaderTransportError('invalid_completion_envelope')
                return httpx.Response(200,json=result,request=request)
            if status in {'failed','unknown'}:
                raise ReaderTransportError('job_'+status+':'+str(row.get('error','unspecified')))
            if status not in {'queued','running'}:
                raise ReaderTransportError('invalid_job_status')
            if self.clock()>=end:
                raise ReaderTransportError('job_wait_deadline_outcome_unknown')
            self.sleep(min(self.poll_s,max(0,end-self.clock())))
            try:
                response=self._request('GET','/v1/reader/jobs/'+job_id,end)
            except httpx.TransportError:
                continue
            if response.status_code in {502,503,504,524}:
                continue
            if response.status_code!=200:
                raise ReaderTransportError('job_poll_http_'+str(response.status_code))
            row=response.json()


def async_reader_adapter_class():
    # Lazy imports keep transport-only CPU tests independent of model packages.
    from .reader_control import ReaderAdapter
    from .adapters import endpoint
    class AsyncReaderAdapter(ReaderAdapter):
        def __init__(self,*args,transport_options,**kwargs):
            super().__init__(*args,**kwargs)
            if self.binding.provider!='owned_runpod_transformers':
                raise ValueError('Async transport is only for the owned reader worker')
            base,key=endpoint(self.binding)
            self.job_transport=ReaderJobTransport(base,key,**transport_options)
            self.job_http_client=httpx.Client(transport=self.job_transport)
            # Replace only SDK wire transport. Prompt/schema/parser remain identical.
            self.client.root_client=self.client.root_client.with_options(http_client=self.job_http_client,max_retries=0)
            self.client.client=self.client.root_client.chat.completions
        def generate(self,*args,**kwargs):
            try:
                return super().generate(*args,**kwargs)
            finally:
                self.last_observation['async_transport']={'version':'reader_jobs_v2',
                    'worker_instance':self.job_transport.instance,
                    'request_namespace':self.job_transport.namespace,
                    'receipts':list(self.job_transport.receipts)}
                self.job_http_client.close()
    return AsyncReaderAdapter
