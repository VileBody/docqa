"""Opt-in private local diagnostic receipts. Never store hidden reasoning/headers."""
import os,re,time
from pathlib import Path
from .util import write,digest

def public_text(message,limit=16384):
    value=getattr(message,'content','')
    if isinstance(value,list):value='\n'.join(v.get('text','') for v in value if isinstance(v,dict) and v.get('type')=='text')
    if not isinstance(value,str):return ''
    value=re.sub(r'<(?:think|analysis|reasoning)>.*?(?:</(?:think|analysis|reasoning)>|$)','[reasoning omitted]',value,flags=re.S|re.I)
    for key,secret in os.environ.items():
        if ('KEY' in key or 'TOKEN' in key or 'SECRET' in key) and len(secret)>=8:value=value.replace(secret,'[REDACTED]')
    return value[:limit]

class DebugReceipt:
    def __init__(self,path,profile,metadata):
        self.path=Path(path);self.profile=profile;self.value={'schema_version':1,'created_at':time.time(),'expires_at':time.time()+profile.debug_retention_hours*3600,
            'classification':profile.data_classification,'access_policy':profile.debug_access_policy,'metadata':metadata,'events':[]}
    def capture(self,event,**data):
        if not self.profile.debug_artifacts:return
        import json
        raw=json.dumps(data,ensure_ascii=False,default=str)
        safe=public_text(type('Public',(),{'content':raw})(),262144)
        try:clean=json.loads(safe)
        except ValueError:clean={'oversize_or_redacted':True,'public_preview':safe[:16384],'original_bytes':len(raw.encode()),'raw_hash':digest(raw.encode())}
        self.value['events'].append({'event':event,**clean})
        self.value['events']=self.value['events'][-8:]
        self.path.parent.mkdir(parents=True,exist_ok=True);self.path.parent.chmod(0o700)
        write(self.path,self.value);self.path.chmod(0o600)

def prune_expired(directory,now=None):
    import json
    now=time.time() if now is None else now;removed=[]
    for p in Path(directory).glob('*.json'):
        d=json.loads(p.read_text())
        if d.get('schema_version')==1 and d.get('expires_at',float('inf'))<=now:p.unlink();removed.append(p.name)
    return removed
