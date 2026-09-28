"""Durable conservative pre-call reservations, serialized across local processes."""
import fcntl,math,time
from contextlib import contextmanager
from pathlib import Path
from .util import BudgetExceeded,read,write

class Budget:
    def __init__(self,path,max_usd=0,max_calls=0,deadline_s=180):
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
        if any(not math.isfinite(v) or v<0 for v in [max_usd,max_calls,deadline_s]):raise ValueError('Invalid budget')
        with self._lock():
            if self.path.exists():
                self.state=read(self.path)
                if self.state['max_usd']!=max_usd or self.state['max_calls']!=max_calls:
                    raise ValueError('Resume budget differs; explicitly authorize a new session')
            else:
                self.state={'max_usd':max_usd,'max_calls':max_calls,'deadline_epoch':time.time()+deadline_s,'reserved_usd':0.,'calls':[]}
                write(self.path,self.state)
    @contextmanager
    def _lock(self):
        with self.path.with_suffix(self.path.suffix+'.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            try:yield
            finally:fcntl.flock(lock,fcntl.LOCK_UN)
    def reserve(self,kind,upper_usd):
        if not math.isfinite(upper_usd) or upper_usd<0:raise ValueError('Invalid reservation')
        with self._lock():
            self.state=read(self.path);s=self.state
            if time.time()>=s['deadline_epoch'] or len(s['calls'])>=s['max_calls'] or s['reserved_usd']+upper_usd>s['max_usd']+1e-12:
                raise BudgetExceeded('Session call/money/deadline limit reached before request')
            s['reserved_usd']+=upper_usd
            s['calls'].append({'kind':kind,'reserved_usd':upper_usd,'started_at':time.time(),'status':'reserved'})
            write(self.path,s);return len(s['calls'])-1
    def complete(self,i,status,usage=None):
        with self._lock():
            self.state=read(self.path)
            self.state['calls'][i].update(status=status,usage=usage,finished_at=time.time());write(self.path,self.state)
    def remaining_s(self):return max(0,self.state['deadline_epoch']-time.time())
