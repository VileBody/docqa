"""Durable Redis catalog/outbox. One fenced atomic publication pointer.

Redis is a single local AOF instance, not a multi-master consensus system.
Celery result TTL is irrelevant to catalog and publication retention.
"""
import json,time,uuid,threading
from contextlib import contextmanager
import redis

class Busy(RuntimeError):pass
class LeaseLost(RuntimeError):pass

class Registry:
    def __init__(self,settings):
        self.settings=settings;self.prefix=settings.namespace+':';self.r=redis.Redis.from_url(settings.redis_url,decode_responses=True,socket_timeout=5,socket_connect_timeout=5)
    def key(self,name):return self.prefix+name
    def get(self,name):
        value=self.r.get(self.key(name));return json.loads(value) if value else None
    def catalog(self):
        return [json.loads(v) for _,v in sorted(self.r.hgetall(self.key('catalog')).items())]
    def document(self,doc_id):
        value=self.r.hget(self.key('catalog'),doc_id);return json.loads(value) if value else None
    def task(self,task_id):return self.get('task:'+task_id)
    def active(self):return self.get('active')
    def register(self,record):
        script='''if redis.call('HEXISTS',KEYS[1],ARGV[1])==1 then return 0 end
        redis.call('HSET',KEYS[1],ARGV[1],ARGV[2]);redis.call('SET',KEYS[2],ARGV[2]);redis.call('SADD',KEYS[3],ARGV[3]);return 1'''
        return bool(self.r.eval(script,3,self.key('catalog'),self.key('task:'+record['task_id']),self.key('pending'),record['doc_id'],json.dumps(record),record['task_id']))
    def pending(self):return sorted(self.r.smembers(self.key('pending')))
    @contextmanager
    def lease(self):
        token=uuid.uuid4().hex;key=self.key('build_lock');ttl=self.settings.lease_seconds*1000
        if not self.r.set(key,token,nx=True,px=ttl):raise Busy('Another generation is being built')
        stop=threading.Event();lost=threading.Event()
        def renew():
            while not stop.wait(max(.2,self.settings.lease_seconds/3)):
                try:
                    ok=self.r.eval("if redis.call('GET',KEYS[1])==ARGV[1] then return redis.call('PEXPIRE',KEYS[1],ARGV[2]) else return 0 end",1,key,token,ttl)
                    if not ok:lost.set();return
                except Exception:lost.set();return
        thread=threading.Thread(target=renew,daemon=True);thread.start()
        try:yield token
        finally:
            stop.set();thread.join(timeout=6)
            self.r.eval("if redis.call('GET',KEYS[1])==ARGV[1] then return redis.call('DEL',KEYS[1]) else return 0 end",1,key,token)
    def update(self,record,token):
        script='''if redis.call('GET',KEYS[1])~=ARGV[1] then return 0 end
        redis.call('HSET',KEYS[2],ARGV[2],ARGV[3]);redis.call('SET',KEYS[3],ARGV[3]);
        if ARGV[4]=='1' then redis.call('SREM',KEYS[4],ARGV[5]) else redis.call('SADD',KEYS[4],ARGV[5]) end;return 1'''
        ok=self.r.eval(script,4,self.key('build_lock'),self.key('catalog'),self.key('task:'+record['task_id']),self.key('pending'),token,record['doc_id'],json.dumps(record),'1' if record['status'] in {'ready','failed'} else '0',record['task_id'])
        if not ok:raise LeaseLost('Stale worker fenced')
    def publish(self,active,records,expected,token):
        # Pointer and catalog updates are atomic ONLY in Redis. Qdrant/files have
        # already been verified, immutable, and are retained for in-flight readers.
        script='''if redis.call('GET',KEYS[1])~=ARGV[1] then return 0 end
        local old=redis.call('GET',KEYS[2]);if old and cjson.decode(old).generation~=ARGV[2] then return 0 end
        if not old and ARGV[2]~='' then return 0 end
        redis.call('SET',KEYS[2],ARGV[3]);local records=cjson.decode(ARGV[4]);
        for _,r in ipairs(records) do local v=cjson.encode(r);redis.call('HSET',KEYS[3],r.doc_id,v);redis.call('SET',ARGV[5]..'task:'..r.task_id,v);redis.call('SREM',KEYS[4],r.task_id) end;return 1'''
        ok=self.r.eval(script,4,self.key('build_lock'),self.key('active'),self.key('catalog'),self.key('pending'),token,expected or '',json.dumps(active),json.dumps(records),self.prefix)
        if not ok:raise LeaseLost('Publication CAS rejected')
