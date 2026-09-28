from celery import Celery
from .config import Settings
from .registry import Registry,Busy,LeaseLost
from .core import Service

settings=Settings.environment()
app=Celery('docqa_stage2',broker=settings.redis_url,backend=settings.redis_url)
app.conf.update(task_acks_late=True,task_reject_on_worker_lost=True,worker_prefetch_multiplier=1,
    task_default_queue=settings.namespace,
    task_serializer='json',result_serializer='json',accept_content=['json'],result_expires=60,
    task_soft_time_limit=840,task_time_limit=900,broker_connection_timeout=5,
    broker_transport_options={'visibility_timeout':960,'socket_timeout':5,'socket_connect_timeout':5},
    result_backend_transport_options={'visibility_timeout':960},visibility_timeout=960,
    beat_schedule={'durable-outbox':{'task':'docqa.dispatch','schedule':10.0}})

@app.task(name='docqa.index',bind=True,max_retries=12)
def index_document(self,task_id):
    try:return Service(Settings.environment()).process(task_id)
    except (Busy,LeaseLost) as exc:raise self.retry(exc=exc,countdown=min(30,2**min(self.request.retries,5)))

def dispatch_one(registry,task_id):
    key=registry.key('dispatch:'+task_id)
    if not registry.r.set(key,'1',nx=True,ex=20):return False
    try:index_document.apply_async(args=[task_id],retry=False);return True
    except Exception:
        registry.r.delete(key)
        return False  # durable outbox remains; beat retries broker publication

@app.task(name='docqa.dispatch')
def dispatch_pending():
    registry=Registry(Settings.environment())
    return sum(dispatch_one(registry,task_id) for task_id in registry.pending())
