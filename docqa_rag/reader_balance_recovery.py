"""Bounded transient balance-API recovery; never authorize calls from stale credit."""
import time
import httpx
from .reader_balance import BalanceMonitor, BalanceUnverified
from .util import read,write


def fetch_balance_with_retries(fetch, *, attempts=6, delay_s=10, sleep=time.sleep, on_error=lambda row:None):
    for attempt in range(attempts):
        try:
            return fetch()
        except (httpx.TransportError, httpx.HTTPStatusError, BalanceUnverified) as exc:
            code=exc.response.status_code if isinstance(exc,httpx.HTTPStatusError) else None
            transient=code is None or code in {408,429,500,502,503,504,524}
            on_error({'attempt':attempt+1,'error_type':type(exc).__name__,'http_status':code,'transient':transient,'at':time.time()})
            if not transient or attempt+1>=attempts:raise
            sleep(delay_s)
    raise ValueError('Positive attempt count required')


class RecoveringBalanceMonitor(BalanceMonitor):
    def __init__(self,path,**kwargs):
        fetch=kwargs.pop('fetch',None) or BalanceMonitor._fetch
        self.retry_path=path.parent/(path.stem+'_RETRIES.json')
        def record(row):
            rows=read(self.retry_path) if self.retry_path.exists() else []
            rows.append(row);write(self.retry_path,rows)
        super().__init__(path,fetch=lambda:fetch_balance_with_retries(fetch,on_error=record),**kwargs)
