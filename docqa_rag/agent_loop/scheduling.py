"""Bounded parallel independent episodes; each episode remains sequential."""
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED

def bounded_jobs(jobs,execute,on_result,can_start,workers=2):
    if workers not in (1,2):raise ValueError('At most two concurrent episodes')
    submitted=0;completed=0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        active={}
        while active or submitted<len(jobs):
            while len(active)<workers and submitted<len(jobs) and can_start():
                job=jobs[submitted];active[pool.submit(execute,job)]=job;submitted+=1
            if not active:break
            done,_=wait(active,return_when=FIRST_COMPLETED)
            for future in done:
                active.pop(future);on_result(future.result());completed+=1
    return {'submitted':submitted,'completed':completed,'unstarted':len(jobs)-submitted}
