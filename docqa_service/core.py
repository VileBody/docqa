"""Idempotent durable ingestion and immutable whole-corpus generations."""
import os,shutil,time,uuid,re
from pathlib import Path
from qdrant_client import QdrantClient
from docqa_rag.util import read,write,digest,ModelUnavailable
from docqa_rag.store import Index,runner_scope
from docqa_rag.sparse import BM25
from docqa_rag.types import Document
from docqa_rag.extraction import extract_source
from docqa_rag.flow import Flow
from docqa_rag.debug import DebugReceipt
from .registry import Registry, Busy, LeaseLost
from .models import create,embedding
from .qdrant import ReadRetryClient

class NotReady(RuntimeError):pass
class Missing(KeyError):pass

def bundle_checks(path,allow_partial=False):
    path=Path(path)
    if path.name.endswith('.partial') and not allow_partial:raise ValueError('Unpublished partial export')
    if digest((path/'checksums.json').read_bytes())!=(path/'checksums.sha256').read_text().strip():
        raise ValueError('Bundle manifest mismatch')
    checks=read(path/'checksums.json')
    actual={str(p.relative_to(path)) for p in path.rglob('*') if p.is_file() and p.name not in {'checksums.json','checksums.sha256'}}
    if actual!=set(checks):raise ValueError('Bundle file set mismatch')
    for name,sha in checks.items():
        raw=path/name;p=raw.resolve()
        if not p.is_relative_to(path.resolve()) or raw.is_symlink() or digest(p.read_bytes())!=sha:
            raise ValueError('Bundle source/checksum mismatch')
    publication=read(path/'publication.json');runner_scope(path/'runner')
    if not re.fullmatch('[0-9a-f]{32}',publication['generation']):raise ValueError('Invalid generation ID')
    for record in publication['records']:
        if not re.fullmatch(r'files/[0-9a-f]{64}\.(txt|pdf)',record['file']):raise ValueError('Invalid source path')
    return publication

class Service:
    def __init__(self,settings):
        self.settings=settings;self.registry=Registry(settings);self.profile=settings.profile();self.selected_reader=settings.reader()
        self.root=settings.data.resolve();self.root.mkdir(parents=True,exist_ok=True)
        self.pipeline=settings.pipeline()
    def client(self):return ReadRetryClient(url=self.settings.qdrant_url,timeout=30)
    def upload(self,data,filename,classification):
        if classification not in {'synthetic','public'}:raise ValueError('Only declared synthetic/public sources allowed')
        suffix=Path(filename).suffix.lower()
        if suffix not in ({'.txt','.pdf'} if self.settings.allow_pdf else {'.txt'}):raise ValueError('TXT required; PDF adapter is opt-in')
        if not data or len(data)>30*1024*1024:raise ValueError('Upload outside 1 byte..30 MiB')
        sha=digest(data);doc_id=digest([sha,suffix,self.pipeline]);task_id=digest(['index',doc_id,self.pipeline])
        path=self.root/'files'/(sha+suffix);path.parent.mkdir(exist_ok=True)
        temp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.partial')
        with temp.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
        os.replace(temp,path)
        record={'doc_id':doc_id,'task_id':task_id,'title':Path(filename).name[:256],
            'file':str(path.relative_to(self.root)),'source_sha256':sha,'source_format':suffix[1:],
            'classification':classification,'pipeline':self.pipeline,'status':'queued','created_at':time.time(),
            'error':None,'attempts':0}
        self.registry.register(record)
        return self.registry.document(doc_id)
    def list_documents(self):return self.registry.catalog()
    def task(self,task_id):
        record=self.registry.task(task_id)
        if record is None:raise Missing(task_id)
        return record
    def retry(self,task_id):
        with self.registry.lease() as token:
            record=self.task(task_id)
            if record['pipeline']!=self.pipeline:raise ValueError('Task pipeline mismatch')
            if record['status']=='ready':return record
            record.update(status='queued',error=None)
            self.registry.update(record,token);return record
    def open_index(self,active):
        if active is None:raise NotReady('No published generation')
        if active['pipeline']!=self.pipeline:raise NotReady('Pipeline mismatch; explicit restore/rebuild needed')
        root=self.root/'snapshots'/active['generation'];bundle_checks(root)
        m=read(root/'index/manifest.json')
        if m['fingerprint']!=active['fingerprint']:raise ValueError('Published fingerprint mismatch')
        obj=Index(self.client(),m['spec'],{k:Document.model_validate(v) for k,v in read(root/'index/documents.json').items()},BM25(state=read(root/'index/bm25.json')),active['collection'])
        try:obj.verify(m['count'])
        except BaseException:obj.close();raise
        return obj
    def fault(self,stage):
        # Used only by real process-kill integration tests, never in production.
        if self.settings.test_mode and os.getenv('DOCQA_TEST_PAUSE_AT')==stage:
            marker=self.root/('fault-'+stage)
            try:
                with marker.open('x') as f:f.write(str(os.getpid()))
            except FileExistsError:return
            while not marker.with_suffix('.release').exists():time.sleep(.1)
    def process(self,task_id):
        record=self.task(task_id)
        with self.registry.lease() as token:
            record=self.task(task_id)
            active=self.registry.active()
            if active and record['doc_id'] in active['document_ids']:
                record.update(status='ready',generation=active['generation'],error=None)
                self.registry.update(record,token);return record
            if record['status']=='failed':return record
            if record['pipeline']!=self.pipeline:raise ValueError('Task pipeline mismatch')
            record.update(status='processing',attempts=record['attempts']+1,error=None)
            self.registry.update(record,token)
            old=None;index=None;generation=uuid.uuid4().hex
            staging=self.root/'staging'/generation;runner=staging/'runner';(runner/'documents').mkdir(parents=True)
            try:
                source=self.root/record['file']
                extract_source(source,record['doc_id'],record['source_sha256'])
                records=[self.registry.document(d) for d in active['document_ids']] if active else []
                records.append(record);entries=[]
                for item in records:
                    f=self.root/item['file'];dest='documents/'+item['doc_id']+f.suffix
                    if digest(f.read_bytes())!=item['source_sha256']:raise ValueError('Immutable upload checksum mismatch')
                    shutil.copyfile(f,runner/dest);entries.append({'id':item['doc_id'],'file':dest,'sha256':item['source_sha256']})
                write(runner/'documents.json',entries)
                write(runner/'run_manifest.json',{'contains_gold':False,'document_count':len(entries),'dataset_hash':digest(entries),
                    'source_titles':{r['doc_id']:r['title'] for r in records}})
                (runner/'questions.jsonl').write_text('')
                embedder=embedding(self.settings)
                collection='docqa_'+self.settings.namespace+'_'+generation
                if active:
                    old=self.open_index(active)
                    index=old.rebuild_snapshot(runner,self.profile,embedder,self.client(),collection)
                else:index=Index.build(runner,self.profile,embedder,self.client(),collection)
                self.fault('after_upsert')
                index.export(staging/'index')
                ready=[{**r,'status':'ready','generation':generation,'error':None} for r in records]
                publication={'generation':generation,'collection':collection,'fingerprint':index.fingerprint,
                    'pipeline':self.pipeline,'document_ids':[r['doc_id'] for r in ready],
                    'records':ready,'metrics':index.metrics,'backend':'test_doubles' if self.settings.test_mode else 'remote_models'}
                write(staging/'publication.json',publication)
                write(staging/'checksums.json',{str(p.relative_to(staging)):digest(p.read_bytes()) for p in staging.rglob('*') if p.is_file()})
                (staging/'checksums.sha256').write_text(digest((staging/'checksums.json').read_bytes())+'\n')
                bundle_checks(staging)
                final=self.root/'snapshots'/generation;final.parent.mkdir(exist_ok=True);staging.rename(final)
                self.fault('before_publish')
                self.registry.publish(publication,ready,active['generation'] if active else '',token)
                self.fault('after_publish')
                return self.registry.document(record['doc_id'])
            except (Busy,LeaseLost):raise
            except Exception as exc:
                current=self.registry.active()
                if current and record['doc_id'] in current['document_ids']:
                    return self.registry.document(record['doc_id'])
                record.update(status='failed',error=type(exc).__name__)
                self.registry.update(record,token)
                return record
            finally:
                if old:old.close()
                if index:index.close()
    def answer(self,doc_id,question,generator=None,diagnostics=False,query_policy='raw',planner_family='qwen'):
        record=self.registry.document(doc_id)
        if record is None:raise Missing(doc_id)
        active=self.registry.active()
        if not active or doc_id not in active['document_ids']:raise NotReady('Document not published')
        index=self.open_index(active)
        try:
            index.diagnostic_replay_enabled=diagnostics
            embedding,reranker,gen=create(self.settings,generator)
            from .submission import make_flow
            planner=gen if planner_family==generator else create(self.settings,planner_family)[2] if query_policy not in {'raw','anchor'} else None
            flow=make_flow(self.settings,index,embedding,reranker,gen,query_policy=query_policy,planner=planner)
            request_id=uuid.uuid4().hex
            flow.debug=DebugReceipt(self.root/'debug'/(request_id+'.json'),self.profile,{'generation':active['generation'],'request_id':request_id})
            answer,trace=flow.ask(doc_id,question)
            result=answer.model_dump()
            for c in result['citations']:
                c['source_sha256']=c.pop('pdf_sha256');c['source_format']=record['source_format']
            result.update(request_id=request_id,generation=active['generation'],fingerprint=index.fingerprint,
                backend=active['backend'],shadow={'mode':flow.critic_mode,'checks':trace.get('checks',[])})
            if diagnostics:
                result['diagnostics']={'limits':{'prefetch_each':self.profile.candidate_limit,'fused_candidates':self.profile.candidate_limit,'rerank_keep':self.profile.rerank_limit,'pack_chars':self.profile.pack_chars},
                    'counts':{'fused_candidates':len(trace['candidates']),'reranker_scored':len(trace['reranked']),'context_sources':len(trace['pack']),'citations':len(answer.citations)},
                    'query_cost':trace['query_cost'],'prefetch_actual_counts':index.branch_diagnostics['branches'],
                    'prefetch_diagnostics':index.branch_diagnostics,
                    'prefetch_note':'Separate branch replay on the same immutable snapshot; primary vectors reused. Not original fusion execution counters.',
                    'candidates':trace['reranked'],'selection_events':trace['selection_events'],'pack_source_ids':[h['chunk']['source_id'] for h in trace['pack']],
                    'cited_source_ids':list(dict.fromkeys(c.source_id for c in answer.citations))}
            write(self.root/'requests'/(request_id+'.json'),{'result':result,'trace':trace,'pipeline':self.pipeline})
            return result
        finally:index.close()
    def export(self,destination):
        active=self.registry.active();index=self.open_index(active);index.close()
        destination=Path(destination)
        if destination.exists():raise FileExistsError(destination)
        temp=destination.with_name(destination.name+'.partial')
        if temp.exists():raise FileExistsError(temp)
        source=self.root/'snapshots'/active['generation'];temp.mkdir(parents=True)
        for i,p in enumerate(sorted(p for p in source.rglob('*') if p.is_file())):
            target=temp/p.relative_to(source);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
            if i==0:self.fault('export_partial')
        bundle_checks(temp,allow_partial=True);temp.rename(destination)
        return {'path':str(destination),'fingerprint':active['fingerprint'],'scope':'published_documents_only',
                'unpublished_catalog_ids_not_in_snapshot':[r['doc_id'] for r in self.registry.catalog() if r['doc_id'] not in active['document_ids']]}
    def restore(self,source,expected_fingerprint):
        source=Path(source);m=bundle_checks(source)
        if m['pipeline']!=self.pipeline or m['fingerprint']!=expected_fingerprint:raise ValueError('Restore pipeline/fingerprint mismatch')
        with self.registry.lease() as token:
            if self.registry.active() or self.registry.catalog():raise ValueError('Restore requires an empty registry namespace')
            generation=uuid.uuid4().hex;collection='docqa_'+self.settings.namespace+'_'+generation
            obj=Index.restore(source/'index',expected_fingerprint,self.client(),collection)
            try:
                final=self.root/'snapshots'/m['generation']
                if final.exists():bundle_checks(final)
                else:final.parent.mkdir(parents=True,exist_ok=True);shutil.copytree(source,final)
                entries=read(source/'runner/documents.json');by_id={r['doc_id']:r for r in m['records']}
                for entry in entries:
                    record=by_id[entry['id']];target=self.root/record['file'];target.parent.mkdir(parents=True,exist_ok=True)
                    shutil.copyfile(source/'runner'/entry['file'],target)
                active={**m,'collection':collection}
                self.registry.publish(active,m['records'],'',token)
                return active
            finally:obj.close()
