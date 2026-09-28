"""Write-ahead request journal. An uncertain request is NEVER silently resubmitted."""
import fcntl, json, os, tempfile
from pathlib import Path
from ..util import digest, ModelUnavailable, ModelOutputInvalid, BudgetExceeded

class PendingRequest(RuntimeError): pass
class IncompatibleCache(ValueError): pass

def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(dir=path.parent,prefix='.'+path.name)
    try:
        os.fchmod(fd,0o600)
        with os.fdopen(fd,'w') as f:
            json.dump(value,f,ensure_ascii=False,sort_keys=True,indent=2);f.flush();os.fsync(f.fileno())
        os.replace(name,path)
        d=os.open(path.parent,os.O_RDONLY)
        try: os.fsync(d)
        finally: os.close(d)
    finally:
        if os.path.exists(name):os.unlink(name)

class Journal:
    def __init__(self,path,identity):
        self.path=Path(path);self.path.mkdir(parents=True,exist_ok=True)
        manifest=self.path/'identity.json';value={'version':2,'identity':identity,'sha256':digest(identity)}
        if manifest.exists():
            if json.loads(manifest.read_text())!=value: raise IncompatibleCache('Journal identity changed')
        else: atomic(manifest,value)
        self.identity=value;self.hits=0
    def call(self,operation,payload,fn,annotations=None):
        # operation is stable episode/call position, payload contains complete wire identity.
        key=digest({'operation':operation,'identity':self.identity['sha256']})
        path=self.path/(key+'.json')
        with (self.path/(key+'.lock')).open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            fingerprint=digest(payload)
            if path.exists():
                row=json.loads(path.read_text())
                if row['input_sha256']!=fingerprint:raise IncompatibleCache('Operation input differs')
                if row['status']=='pending':raise PendingRequest('Uncertain request; inspect ledger/provider before reconciliation: '+operation)
                self.hits+=1
                if row['status']=='failed':
                    cls={'ModelOutputInvalid':ModelOutputInvalid,'BudgetExceeded':BudgetExceeded}.get(row['error_type'],ModelUnavailable)
                    raise cls('Saved failure; not resampled: '+row['error_type'])
                return row['result']
            row={'operation':operation,'input_sha256':fingerprint,'input':payload,'status':'pending'}
            atomic(path,row)
            try: result=fn()
            except Exception as exc:
                # Network exceptions can represent executed requests. Persist failure, never retry automatically.
                atomic(path,{**row,'status':'failed','error_type':type(exc).__name__});raise
            atomic(path,{**row,'status':'completed','result':result,**({'provenance':annotations()} if annotations else {})});return result

class SharedResponses:
    """Exact full-input reuse across independent controller/reader/shadow namespaces.

    Raw responses remain unchanged. Failures and uncertain writes retain the
    same no-resubmission policy as ordinary journal calls.
    """
    def __init__(self,path):
        self.journal=Journal(path,{'contract':'exact-full-model-wire-input-v2'})
    @staticmethod
    def payload(value):
        return {k:value[k] for k in ('kind','request','output_reserve')}
    def call(self,value,fn,annotations=None):
        value=self.payload(value);before=self.journal.hits
        result=self.journal.call(digest(value),value,fn,annotations)
        return result,self.journal.hits>before,digest(value)
    def seed(self,request_root):
        count=0
        for p in sorted(Path(request_root).rglob('*.json'),key=lambda p:p.stat().st_mtime_ns):
            if p.name=='identity.json':continue
            row=json.loads(p.read_text())
            if row.get('status')!='completed' or not row.get('operation','').startswith(('episode/','reader/','shadow/','shadow-replay/','replay/')):continue
            if row['input']['kind'] not in {'qwen','openai'}:continue
            # Seeding preserves the first completed response, never selects by quality.
            self.call(row['input'],lambda r=row:r['result'],lambda p=p:{'seeded_from':str(p)})
            count+=1
        return count
