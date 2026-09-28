"""One collection per compatible index; Qdrant does native dense/BM25 fusion."""
from __future__ import annotations
import importlib.metadata, shutil, time, uuid
from pathlib import Path
from qdrant_client import QdrantClient, models as qm
from .types import Chunk, Document, Hit, Profile
from .extraction import extract_source, split_document, PARSER, TEXT_PARSER
from .sparse import BM25
from .util import digest, read, write

COLLECTION = 'docqa'
PAYLOAD_INDEXES = {'doc_id':'keyword','snapshot':'keyword','ready':'bool'}

def runner_scope(path):
    root=Path(path).resolve(); manifest=read(root/'run_manifest.json')
    if manifest.get('contains_gold') is not False: raise ValueError('Require safe runner export')
    docs=read(root/'documents.json'); ids=set()
    for d in docs:
        if set(d)!={'id','file','sha256'} or d['id'] in ids: raise ValueError('Invalid document contract')
        p=(root/d['file']).resolve()
        if not p.is_relative_to(root/'documents') or p.suffix.lower() not in {'.pdf','.txt'}: raise ValueError('Source outside scope')
        if digest(p.read_bytes()) != d['sha256']: raise ValueError('Source hash mismatch')
        ids.add(d['id'])
    if manifest['document_count']!=len(docs): raise ValueError('Document count mismatch')
    return manifest,docs

def index_spec(path,profile):
    manifest,docs=runner_scope(path)
    binding=profile.embedding.model_dump(exclude={'query_instruction','endpoint_env','key_env','live_smoke','chat_template_sha256','reasoning','provider','origin_verification','model_input_budget','model_context_budget','max_output_tokens','tokenizer_path','context_limit_source'})
    spec={'dataset_hash':manifest['dataset_hash'],'scope':docs,'assembly':'single_file',
          'parser':PARSER if all(Path(d['file']).suffix.lower()=='.pdf' for d in docs) else {'pdf':PARSER,'txt':TEXT_PARSER},'splitter':{'name':'RecursiveCharacterTextSplitter','version':importlib.metadata.version('langchain-text-splitters'),
            'size':profile.chunk_size,'overlap':profile.chunk_overlap,'page_boundaries':True,'normalization':'none'},
          'embedding':binding,'bm25':{'tokenizer':'unicode-word-or-identifier-casefold-v1','idf':'frozen-corpus','k1':1.2,'b':0.75},
          'qdrant_client':importlib.metadata.version('qdrant-client'),'numeric_mode':'float32-cosine',
          'payload_indexes':PAYLOAD_INDEXES,'schema_version':1}
    if manifest.get('source_titles'):
        spec['source_titles']={d['id']:manifest['source_titles'].get(d['id'],d['id']) for d in docs}
    return spec

class Index:
    diagnostic_replay_enabled = False

    def observe_branches(self, prefetch):
        """Separate same-snapshot diagnostic queries, reusing primary vectors.

        These observations do not change fusion, reranking, or model calls.
        They are a replay, not counters from the server's original execution.
        """
        if not self.diagnostic_replay_enabled:
            return
        started = time.perf_counter()
        rows = []
        for branch in prefetch:
            result = self.client.query_points(self.collection, query=branch.query,
                using=branch.using, query_filter=branch.filter, limit=branch.limit,
                with_payload=False, with_vectors=False)
            rows.append({'branch':branch.using,'count':len(result.points),'limit':branch.limit})
        self.branch_diagnostics = {'status':'observed','method':'separate_branch_replay',
            'snapshot':self.fingerprint,'branches':rows,'extra_qdrant_requests':len(rows),
            'extra_model_calls':0,'elapsed_s':time.perf_counter()-started}

    def __init__(self, client: QdrantClient, spec: dict, documents: dict[str,Document], bm25: BM25, collection=COLLECTION):
        self.client=client; self.collection=collection; self.spec=spec; self.fingerprint=digest(spec); self.documents=documents; self.bm25=bm25; self.metrics={}

    @classmethod
    def build(cls, runner, profile: Profile, embedder, client=None, collection=COLLECTION):
        client=client or QdrantClient(':memory:'); spec=index_spec(runner,profile)
        if client.collection_exists(collection): raise ValueError('Collection exists; use validated restore/reuse or fresh instance')
        start=time.perf_counter(); docs={}; chunks=[]
        for entry in spec['scope']:
            doc=extract_source(Path(runner)/entry['file'],entry['id'],entry['sha256'])
            doc.doc_title=spec.get('source_titles',{}).get(doc.doc_id,doc.doc_id)
            docs[doc.doc_id]=doc; chunks+=split_document(doc,profile.chunk_size,profile.chunk_overlap)
        extraction=time.perf_counter()-start
        t=time.perf_counter(); bm25=BM25([c.text for c in chunks]); sparse=[bm25.vector(c.text) for c in chunks]
        sparse_s=time.perf_counter()-t; t=time.perf_counter()
        dense=embedder.embed_documents([c.text for c in chunks]); dense_s=time.perf_counter()-t
        dimension=profile.embedding.dimension
        if len(dense)!=len(chunks) or any(len(v)!=dimension for v in dense): raise ValueError('Embedding count/dimension mismatch')
        obj=cls(client,spec,docs,bm25,collection); obj._create(dimension); t=time.perf_counter()
        points=[qm.PointStruct(id=str(uuid.uuid5(uuid.NAMESPACE_URL,c.doc_id+':'+c.source_id)),
            vector={'dense':v,'bm25':s},payload={**c.model_dump(),'snapshot':obj.fingerprint,'ready':True}) for c,v,s in zip(chunks,dense,sparse)]
        try:
            for i in range(0,len(points),64): client.upsert(collection,points[i:i+64],wait=True)
            obj.verify(len(points))
        except BaseException:
            client.delete_collection(collection); raise
        obj.metrics={'extraction_split_s':extraction,'bm25_s':sparse_s,'dense_s':dense_s,
            'upsert_readiness_s':time.perf_counter()-t,'cold_build_s':time.perf_counter()-start,'points':len(points)}
        return obj

    def rebuild_snapshot(self,runner,profile,embedder,client=None,collection=COLLECTION):
        """Stage1.1 offline publication into a NEW empty instance, never in-place.

        Frozen sparse statistics are rebuilt; dimensions append, old dense vectors
        are reused only for identical text and embedding semantic bindings.
        A service/atomic pointer switch remains a Stage2 gate.
        """
        target=client or QdrantClient(':memory:')
        if target.collection_exists(collection):raise ValueError('Publish rebuilt snapshot only to an empty isolated instance')
        spec=index_spec(runner,profile);documents={};chunks=[]
        for entry in spec['scope']:
            document=extract_source(Path(runner)/entry['file'],entry['id'],entry['sha256'])
            document.doc_title=spec.get('source_titles',{}).get(document.doc_id,document.doc_id)
            documents[document.doc_id]=document;chunks+=split_document(document,profile.chunk_size,profile.chunk_overlap)
        old_binding=self.spec['embedding'];new_binding=spec['embedding']
        reusable={p.payload['text']:p.vector['dense'] for p in self.points()} if old_binding==new_binding else {}
        missing=list(dict.fromkeys(c.text for c in chunks if c.text not in reusable))
        if missing:
            vectors=embedder.embed_documents(missing)
            if len(vectors)!=len(missing):raise ValueError('Embedding count mismatch')
            reusable.update(zip(missing,vectors))
        bm25=BM25([c.text for c in chunks],previous_vocabulary=self.bm25.state['vocabulary'])
        spec['bm25'].update(policy='frozen-rebuild-append-vocabulary-v2',vocabulary_sha256=digest(bm25.state['vocabulary']),
            statistics_sha256=digest(bm25.state),statistics_corpus=digest(spec['scope']),idf_location='query_only',qdrant_modifier=None)
        obj=Index(target,spec,documents,bm25,collection);obj._create(profile.embedding.dimension)
        try:
            points=[qm.PointStruct(id=str(uuid.uuid5(uuid.NAMESPACE_URL,c.doc_id+':'+c.source_id)),
                vector={'dense':reusable[c.text],'bm25':bm25.vector(c.text)},payload={**c.model_dump(),'snapshot':obj.fingerprint,'ready':True}) for c in chunks]
            for i in range(0,len(points),64):target.upsert(collection,points[i:i+64],wait=True)
            obj.verify(len(points))
        except BaseException:target.delete_collection(collection);raise
        obj.metrics={'points':len(points),'dense_new_texts':len(missing),'dense_reused_chunks':sum(c.text not in missing for c in chunks),
            'publication':'new isolated snapshot verified; caller publishes its pointer','parent_fingerprint':self.fingerprint}
        return obj

    def _create(self,dimension):
        self.client.create_collection(self.collection,vectors_config={'dense':qm.VectorParams(size=dimension,distance=qm.Distance.COSINE)},
            sparse_vectors_config={'bm25':qm.SparseVectorParams()})
        # Embedded Qdrant supports filters but ignores physical indexes; server creates these indexes.
        if not isinstance(self.client._client, __import__('qdrant_client.local.qdrant_local',fromlist=['QdrantLocal']).QdrantLocal):
            for name,kind in PAYLOAD_INDEXES.items(): self.client.create_payload_index(self.collection,name,kind,wait=True)

    def points(self):
        result=[]; offset=None
        while True:
            records,offset=self.client.scroll(self.collection,limit=128,offset=offset,with_payload=True,with_vectors=True)
            result.extend(records)
            if offset is None: return result

    def verify(self,count):
        info=self.client.get_collection(self.collection)
        if info.points_count != count: raise ValueError('Point count mismatch')
        vectors=info.config.params.vectors; sparse=info.config.params.sparse_vectors
        if set(vectors)!={'dense'} or vectors['dense'].size != self.spec['embedding']['dimension'] or set(sparse or {})!={'bm25'}:
            raise ValueError('Vector schema mismatch')
        for p in self.points():
            payload=p.payload
            if payload.get('snapshot')!=self.fingerprint or payload.get('ready') is not True: raise ValueError('Unpublished/stale point')
            c=Chunk.model_validate({k:v for k,v in payload.items() if k not in {'snapshot','ready'}})
            doc=self.documents.get(c.doc_id)
            if not doc or doc.text[c.char_start:c.char_end]!=c.text or c.text_sha256!=doc.text_sha256 or c.pdf_sha256!=doc.pdf_sha256:
                raise ValueError('Corrupt point provenance')

    def search(self,doc_id,query,embedder,limit=24,branch='hybrid',fusion='rrf'):
        if doc_id not in self.documents: raise ValueError('Document outside ready scope')
        filt=qm.Filter(must=[qm.FieldCondition(key='doc_id',match=qm.MatchValue(value=doc_id)),
            qm.FieldCondition(key='snapshot',match=qm.MatchValue(value=self.fingerprint)),
            qm.FieldCondition(key='ready',match=qm.MatchValue(value=True))])
        if branch not in {'hybrid','dense','bm25'}: raise ValueError('Unknown branch')
        sparse=self.bm25.vector(query.lexical,query=True)
        dense=embedder.embed_query(query.dense) if branch!='bm25' else None
        if branch=='hybrid':
            result=self.client.query_points(self.collection,prefetch=[
                qm.Prefetch(query=dense,using='dense',filter=filt,limit=limit),
                qm.Prefetch(query=sparse,using='bm25',filter=filt,limit=limit)],
                query=qm.FusionQuery(fusion=qm.Fusion(fusion)),query_filter=filt,limit=limit,with_payload=True)
        else:
            result=self.client.query_points(self.collection,query=dense if branch=='dense' else sparse,
                using='dense' if branch=='dense' else 'bm25',query_filter=filt,limit=limit,with_payload=True)
        return [Hit(chunk=Chunk.model_validate({k:v for k,v in p.payload.items() if k not in {'snapshot','ready'}}),retrieval_score=p.score) for p in result.points]

    def search_views(self,doc_id,views,embedder,limit=24,fusion='rrf'):
        if doc_id not in self.documents:raise ValueError('Document outside ready scope')
        if not 1<=len(views.dense_views)<=4:raise ValueError('View budget exceeded')
        filt=qm.Filter(must=[qm.FieldCondition(key='doc_id',match=qm.MatchValue(value=doc_id)),
            qm.FieldCondition(key='snapshot',match=qm.MatchValue(value=self.fingerprint)),
            qm.FieldCondition(key='ready',match=qm.MatchValue(value=True))])
        prefetch=[qm.Prefetch(query=embedder.embed_query(v),using='dense',filter=filt,limit=limit) for v in views.dense_views]
        prefetch.append(qm.Prefetch(query=self.bm25.vector(views.lexical,query=True),using='bm25',filter=filt,limit=limit))
        result=self.client.query_points(self.collection,prefetch=prefetch,query=qm.FusionQuery(fusion=qm.Fusion(fusion)),query_filter=filt,limit=limit,with_payload=True)
        self.observe_branches(prefetch)
        return [Hit(chunk=Chunk.model_validate({k:v for k,v in p.payload.items() if k not in {'snapshot','ready'}}),retrieval_score=p.score) for p in result.points]

    def export(self,destination):
        destination=Path(destination)
        if destination.exists(): raise FileExistsError(destination)
        temp=destination.with_name(destination.name+'.partial')
        if temp.exists(): raise FileExistsError(temp)
        temp.mkdir(parents=True); t=time.perf_counter()
        try:
            points=[{'id':str(p.id),'payload':p.payload,'vector':p.model_dump(mode='json')['vector']} for p in self.points()]
            write(temp/'points.json',points)
            write(temp/'documents.json',{k:v.model_dump() for k,v in self.documents.items()})
            write(temp/'bm25.json',self.bm25.state)
            checks={f.name:digest(f.read_bytes()) for f in temp.iterdir()}
            write(temp/'manifest.json',{'fingerprint':self.fingerprint,'spec':self.spec,'count':len(points),'checksums':checks,
                'build_metrics':self.metrics,'publication':'ready','format':'qdrant-points-v1','payload_indexes':PAYLOAD_INDEXES,'backend_version':self.client.info().version})
            manifest_sha=digest((temp/'manifest.json').read_bytes()); (temp/'manifest.sha256').write_text(manifest_sha+'\n')
            temp.rename(destination)
        except BaseException:
            shutil.rmtree(temp); raise
        return {'path':str(destination),'manifest_sha256':manifest_sha,'bytes':sum(p.stat().st_size for p in destination.iterdir()),'export_s':time.perf_counter()-t}

    @classmethod
    def restore(cls,source,expected_fingerprint,client=None,collection=COLLECTION):
        source=Path(source); t=time.perf_counter()
        if digest((source/'manifest.json').read_bytes())!=(source/'manifest.sha256').read_text().strip(): raise ValueError('Manifest checksum mismatch')
        m=read(source/'manifest.json')
        if m['fingerprint']!=expected_fingerprint or digest(m['spec'])!=expected_fingerprint or m['publication']!='ready': raise ValueError('Incompatible index')
        if set(m['checksums'])!={'points.json','documents.json','bm25.json'}: raise ValueError('Unexpected snapshot files')
        for name,sha in m['checksums'].items():
            if digest((source/name).read_bytes())!=sha: raise ValueError('Snapshot checksum mismatch')
        client=client or QdrantClient(':memory:')
        if client.collection_exists(collection): raise ValueError('Restore requires empty instance')
        obj=cls(client,m['spec'],{k:Document.model_validate(v) for k,v in read(source/'documents.json').items()},BM25(state=read(source/'bm25.json')),collection)
        for doc in obj.documents.values():
            if digest(doc.text.encode())!=doc.text_sha256: raise ValueError('Text digest mismatch')
        obj._create(m['spec']['embedding']['dimension'])
        try:
            points=[qm.PointStruct.model_validate(p) for p in read(source/'points.json')]
            for i in range(0,len(points),64): client.upsert(collection,points[i:i+64],wait=True)
            obj.verify(m['count'])
            # Fixed lexical smoke after restore, independent of external model availability.
            for doc_id in obj.documents:
                p=next((p for p in points if p.payload['doc_id']==doc_id),None)
                if p is None: raise ValueError('Ready document without points')
                from .flow import plan_query
                if not obj.search(doc_id,plan_query(p.payload['text'][:200]),None,branch='bm25',limit=1): raise ValueError('Restore retrieval smoke failed')
        except BaseException:
            client.delete_collection(collection); raise
        obj.metrics={'restore_s':time.perf_counter()-t,'cold_build_metrics':m['build_metrics'],'points':len(points)}
        return obj

    def close(self): self.client.close()

def cache_decision(policy,*,build_s,build_usd,artifact_bytes,transfer_bytes_per_s,restore_s,expected_reuses,hourly_usd,seconds_value_usd=0,free_bytes=None):
    if min(build_s,build_usd,artifact_bytes,restore_s,expected_reuses,hourly_usd,seconds_value_usd)<0 or transfer_bytes_per_s<=0: raise ValueError('Invalid cache measurements')
    transfer_s=artifact_bytes/transfer_bytes_per_s
    rebuild=expected_reuses*(build_usd+build_s*seconds_value_usd)
    persist=(1+expected_reuses)*transfer_s*(hourly_usd/3600+seconds_value_usd)+expected_reuses*restore_s*(hourly_usd/3600+seconds_value_usd)
    fits=free_bytes is None or artifact_bytes*2<=free_bytes
    chosen=policy if policy!='auto' else ('persist_local' if fits and persist<rebuild else 'rebuild')
    if chosen=='persist_local' and not fits: raise ValueError('Insufficient local disk')
    return {'policy':chosen,'requested':policy,'rebuild_equivalent_usd':rebuild,'persist_equivalent_usd':persist,
            'transfer_s':transfer_s,'expected_reuses':expected_reuses,'fits_disk':fits,'measured_build_s':build_s}
