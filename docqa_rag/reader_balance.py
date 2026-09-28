"""Read-only RunPod balance checks during an authorized reader session."""
import os,time
import httpx
from .util import read,write

class BalanceLow(RuntimeError):pass
class BalanceUnverified(RuntimeError):pass

class BalanceMonitor:
    def __init__(self,path,threshold=.5,interval=60,fetch=None):
        self.path=path;self.threshold=threshold;self.interval=interval;self.fetch=fetch or self._fetch
    @staticmethod
    def _fetch():
        with httpx.Client(timeout=20,follow_redirects=False) as c:
            r=c.post('https://api.runpod.io/graphql',headers={'Authorization':'Bearer '+os.environ['RUNPOD_API_KEY']},
                     json={'query':'query { myself { clientBalance currentSpendPerHr } }'})
            r.raise_for_status();data=r.json()
            value=(data.get('data') or {}).get('myself')
            if value is None:raise BalanceUnverified('RunPod did not expose account balance')
            return value
    def check(self,now=None):
        now=time.time() if now is None else now
        rows=read(self.path) if self.path.exists() else []
        if not rows or now-rows[-1]['checked_at']>=self.interval:
            try:
                value=self.fetch();balance=float(value['clientBalance'])
                row={'checked_at':now,'remaining_usd':balance,'current_spend_per_hour':float(value['currentSpendPerHr']),
                     'threshold_usd':self.threshold,'state':'low' if balance<=self.threshold else 'ok'}
            except Exception as e:
                row={'checked_at':now,'remaining_usd':None,'state':'unverified','error_type':type(e).__name__}
            rows.append(row);write(self.path,rows)
        row=rows[-1]
        if row['state']=='low':raise BalanceLow('RunPod balance is at or below $0.50; checkpoint, notify user, release owned GPU while waiting')
        if row['state']=='unverified':raise BalanceUnverified('Balance unavailable; pause new submissions, do not assume sufficient credit')
        return row
