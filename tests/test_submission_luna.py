"""Product HTTP/core/SDK contract checks; model transport is explicitly simulated."""
import json, sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import httpx,pytest
from fastapi.testclient import TestClient
from langchain_openai import ChatOpenAI
from docqa_rag.util import read,write,digest,ModelOutputInvalid
from docqa_rag.reader_control import ReaderAdapter,request_parts
from docqa_rag.budget import Budget
from docqa_rag.types import Draft,Document,Hit
from docqa_rag.evidence import resolve_claim,EvidenceIssue
from docqa_service.config import Settings,ROOT
from docqa_service.submission import make_flow
from docqa_service.models import TestEmbedding as DoubleEmbedding,TestReranker as DoubleReranker

@pytest.fixture
def settings(tmp_path):
    ledger=tmp_path/'ledger.json';Budget(ledger,1,12,300)
    return replace(Settings.environment(),data=tmp_path/'data',api_key='contract-only-api-key',
        profile_path=ROOT/'profiles/stage_2/txt.json',reader_profile=ROOT/'profiles/submission/luna_p1.json',
        model_calls=True,ledger=str(ledger),test_mode=False)

def pack():
    data=read(ROOT/'examples/acceptance/source_fidelity_v2/faithful_paraphrase.pack.json')
    return Document.model_validate(data['document']),[Hit.model_validate(h) for h in data['pack']]

def mock_factory(settings,monkeypatch,outputs=None):
    import docqa_service.models as m
    import docqa_rag.adapters as a
    b=settings.profile().generators['openai'];sent=[]
    monkeypatch.setenv(b.endpoint_env,'https://fixture.invalid/v1');monkeypatch.setenv(b.key_env,'not-a-live-key')
    results=iter(outputs or [{'status':'not_found','claims':[],'missing':['not established']}]*5)
    def transport(r):
        sent.append(json.loads(r.content));return httpx.Response(200,json={'id':'simulated-wire','object':'chat.completion','created':1,'model':b.model_id,'choices':[{'index':0,'message':{'role':'assistant','content':json.dumps(next(results))},'finish_reason':'stop'}]})
    monkeypatch.setattr(a,'ChatOpenAI',lambda **kwargs:ChatOpenAI(**kwargs,http_client=httpx.Client(transport=httpx.MockTransport(transport))))
    monkeypatch.setattr(m,'QwenEmbeddings',lambda *a:DoubleEmbedding())
    monkeypatch.setattr(m,'InstructionReranker',lambda *a:DoubleReranker())
    return m.create(settings),sent

def test_product_factory_exact_sdk_request_and_strict_projection(settings,monkeypatch):
    (embedding,reranker,reader),sent=mock_factory(settings,monkeypatch)
    assert isinstance(reader,ReaderAdapter)
    doc,hits=pack();q='Что сообщает источник?'
    from docqa_rag.debug import DebugReceipt
    flow=make_flow(settings,SimpleNamespace(documents={doc.doc_id:doc}),embedding,reranker,reader)
    assert flow.answerability_policy=='strict_answerability_v1' and flow.critic_mode=='off'
    flow.debug=DebugReceipt(settings.data/'debug.json',settings.profile(),{})
    answer,_=flow.answer_pack(doc.doc_id,q,hits)
    assert not answer.found
    expected=request_parts(q,hits,reader.prompt_config,ROOT)
    wire=sent[0]
    assert wire['messages']==expected[5] and len(wire['messages'])==14
    kit=ROOT/'incoming/finalize_luna_p1_v1/docqa_finalize_luna_p1/selected_reader/WIRE_RESPONSE_FORMAT.json'
    # Cleanroom ships the exact pinned schema independently of the private instruction kit.
    reference=read(kit if kit.exists() else ROOT/'profiles/submission/WIRE_RESPONSE_FORMAT.json')
    assert wire['response_format']==reference
    assert wire['temperature']==0 and wire['reasoning_effort']=='none'
    assert wire.get('max_tokens',wire.get('max_completion_tokens'))==2048
    assert reader.last_observation['context']['prompt_hash']==digest(expected[5])
    assert reader.last_observation['prior_user_history']==[]
    assert reader.last_observation['teaching_pairs']==6

@pytest.mark.parametrize('status,claims,missing',[
    ('complete',True,['gap']),('complete',False,[]),('partial',True,[]),('partial',False,['gap']),
    ('not_found',True,[]),('ambiguous',True,[]),('contradictory',True,[])])
def test_product_rejects_inconsistent_states(settings,status,claims,missing):
    doc,hits=pack();flow=make_flow(settings,SimpleNamespace(documents={doc.doc_id:doc}),None,None,None)
    value=Draft(status=status,missing=missing,claims=[{'text':'fact','references':[{'source_id':hits[0].chunk.source_id,'span_id':'L1'}]}] if claims else [])
    with pytest.raises(ModelOutputInvalid):flow.validate_draft(doc.doc_id,hits,value)

def test_product_missing_prose_and_source_isolation(settings):
    doc,hits=pack();flow=make_flow(settings,SimpleNamespace(documents={doc.doc_id:doc}),None,None,None)
    valid={'text':hits[0].chunk.text.strip(),'references':[{'source_id':hits[0].chunk.source_id,'span_id':'L1'}]}
    a,_=flow.validate_draft(doc.doc_id,hits,Draft(status='partial',claims=[valid],missing=['Invented 999 fee']))
    assert '999' not in a.answer
    for ref in [{'source_id':'teaching-only','span_id':'L1'}, {'source_id':hits[0].chunk.source_id,'span_id':'L999'}, {'source_id':hits[0].chunk.source_id,'quote':'fabrication','start':0,'end':11}]:
        bad=Draft(status='complete',claims=[{'text':'fact','references':[ref]}],missing=[])
        with pytest.raises(EvidenceIssue):flow.validate_draft(doc.doc_id,hits,bad)
    foreign=hits[0].model_copy(update={'chunk':hits[0].chunk.model_copy(update={'doc_id':'foreign'})})
    with pytest.raises(ValueError):flow.answer_pack(doc.doc_id,'q',[foreign])

def test_reader_change_cannot_invalidate_document_embeddings(settings,monkeypatch):
    before=settings.pipeline()
    monkeypatch.setenv('DOCQA_OPENAI_MODEL','alternative-alias')
    monkeypatch.setenv('DOCQA_OPENAI_REVISION','explicit-alternative')
    monkeypatch.setenv('DOCQA_PACK_CHARS','9000')
    assert settings.pipeline()==before
    monkeypatch.setenv('DOCQA_EMBED_DIMENSION','512')
    assert settings.pipeline()!=before

def test_http_actual_factory_no_prior_question_history(settings,monkeypatch):
    import docqa_service.core as c
    from docqa_service.api import create_app
    from docqa_rag.store import Index
    from docqa_service.models import TestGenerator as DoubleGenerator
    from docqa_rag.extraction import extract_text
    # Real chunking/BM25/Qdrant local Query API; deterministic model transport.
    adapters,sent=mock_factory(settings,monkeypatch)
    tmp=settings.data;tmp.mkdir();runner=tmp/'runner';(runner/'documents').mkdir(parents=True)
    raw=(ROOT/'examples/acceptance/alpha.txt').read_bytes();doc_id='a'*64
    (runner/'documents/a.txt').write_bytes(raw)
    write(runner/'documents.json',[{'id':doc_id,'file':'documents/a.txt','sha256':digest(raw)}])
    write(runner/'run_manifest.json',{'contains_gold':False,'document_count':1,'dataset_hash':digest(raw)})
    (runner/'questions.jsonl').write_text('')
    p=settings.profile().model_copy(deep=True);p.embedding.dimension=8;p.threshold_enabled=False
    idx=Index.build(runner,p,DoubleEmbedding())
    monkeypatch.setattr(Settings,'profile',lambda self:p)
    registry=SimpleNamespace(document=lambda _: {'source_format':'txt'},active=lambda:{'document_ids':[doc_id],'generation':'g','backend':'test_doubles'})
    monkeypatch.setattr(c,'Registry',lambda _:registry)
    monkeypatch.setattr(c.Service,'open_index',lambda *a:idx)
    monkeypatch.setattr(idx,'close',lambda:None)
    monkeypatch.setattr(c,'create',lambda *a:adapters)
    app=TestClient(create_app(settings));auth={'Authorization':'Bearer '+settings.api_key}
    for q in ['Предыдущий вопрос Q1','Срок доставки Q2','Срок доставки Q2']:
        res=app.post('/questions',headers=auth,json={'document_id':doc_id,'question':q,'diagnostics':True})
        assert res.status_code==200,res.text
    assert len(sent)==3 # no generation cache reuse; independent calls
    assert sent[1]['messages']==sent[2]['messages']
    assert 'Предыдущий вопрос Q1' not in json.dumps(sent[1])
    assert all(len(x['messages'])==14 for x in sent)
    assert app.post('/questions',headers=auth,json={'document_id':doc_id,'question':'q','generator':'qwen'}).status_code==422

def test_runtime_does_not_import_evaluator():
    import subprocess
    result=subprocess.run([sys.executable,'-c',"import sys; import docqa_service.api; assert 'docqa_rag.judging' not in sys.modules; assert not any(n.startswith('benchmarks.') for n in sys.modules)"],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
