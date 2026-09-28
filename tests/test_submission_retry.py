"""Sol/P1 v2 retry contract: local MockTransport, no real endpoints or paid calls."""
from dataclasses import replace
import os
import httpx
import pytest
from docqa_service.config import Settings, ROOT
from docqa_service.models import calls
from docqa_rag.budget import Budget
from docqa_rag.util import read, ModelUnavailable, ModelOutputInvalid, BudgetExceeded, InvalidEvidence
from docqa_rag.reader_control import request_parts
from docqa_rag.request_identity import model_request
from docqa_rag.types import Hit

@pytest.fixture(autouse=True)
def no_external_settings(monkeypatch):
    for key in list(os.environ):
        if key.startswith('DOCQA_'):monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr('docqa_rag.adapters.time.sleep', lambda _: None)


def selected(tmp_path, max_usd=1, max_calls=10):
    ledger=tmp_path/'ledger.json'
    Budget(ledger,max_usd,max_calls,600)
    settings=Settings(tmp_path,'','','',ROOT/'profiles/submission/sol_p1_runtime.json',
        reader_profile=ROOT/'profiles/submission/sol_p1_v2.json',model_calls=True,ledger=str(ledger))
    return settings,calls(settings)


def wire(sequence):
    sent=[];values=iter(sequence)
    def handle(request):
        sent.append(request)
        value=next(values)
        if isinstance(value,Exception):raise value
        return httpx.Response(value,json={'ok':value==200},request=request)
    client=httpx.Client(transport=httpx.MockTransport(handle))
    def request():
        response=client.post('https://fixture.invalid/v1/mock',json={'fixture':'fixed'})
        response.raise_for_status()
        return response.json()
    return request,sent


@pytest.mark.parametrize('kind',['openai','embedding','reranker'])
def test_factory_retries_503_then_success_for_every_model_role(tmp_path,kind):
    settings,c=selected(tmp_path);request,sent=wire([503,200])
    assert settings.reader().version=='submission-sol-p1-v2' and c.max_attempts==3
    assert c.invoke(kind,'x',request)=={'ok':True}
    assert len(sent)==2 and sent[0].content==sent[1].content
    assert [x['attempt'] for x in c.records]==[1,2]
    assert [x['status'] for x in read(settings.ledger)['calls']]==['failed','completed']
    assert len(read(settings.ledger)['calls'])==2 # one logical call, two billable reservations


@pytest.mark.parametrize('status',[408,429,500,502,503,504])
def test_classified_transient_statuses(tmp_path,status):
    _,c=selected(tmp_path);request,sent=wire([status,200])
    assert c.invoke('openai','x',request)['ok'] and len(sent)==2


def test_ambiguous_timeout_reserves_and_records_both_attempts(tmp_path):
    settings,c=selected(tmp_path);request,sent=wire([httpx.ReadTimeout('simulated lost response'),200])
    c.invoke('openai','x',request)
    ledger=read(settings.ledger)
    assert len(sent)==2 and len(ledger['calls'])==2
    assert ledger['reserved_usd']==sum(x['reserved_usd'] for x in ledger['calls'])
    assert c.records[0]['error_type']=='ReadTimeout' # not erased or treated as an independent success


@pytest.mark.parametrize('status',[400,401,403,404,422])
def test_permanent_http_error_is_not_retried(tmp_path,status):
    _,c=selected(tmp_path);request,sent=wire([status])
    with pytest.raises(ModelUnavailable):c.invoke('openai','x',request)
    assert len(sent)==1 and len(c.records)==1


@pytest.mark.parametrize('error',[ModelOutputInvalid('invalid JSON'),InvalidEvidence('unissued reference'),ValueError('bad config')])
def test_invalid_output_evidence_and_config_are_not_retried(tmp_path,error):
    settings,c=selected(tmp_path);seen=[]
    def fail():seen.append(1);raise error
    with pytest.raises((ModelOutputInvalid,ModelUnavailable)):c.invoke('openai','x',fail)
    assert len(seen)==1 and len(read(settings.ledger)['calls'])==1


def test_semantic_result_is_returned_without_reroll(tmp_path):
    _,c=selected(tmp_path);seen=[]
    def outcome():seen.append(1);return {'supported':False,'reason':'fixture semantic FAIL'}
    assert c.invoke('openai','x',outcome)['supported'] is False and len(seen)==1


def test_three_transient_failures_stop_at_three_total(tmp_path):
    settings,c=selected(tmp_path);request,sent=wire([503,503,503])
    with pytest.raises(ModelUnavailable):c.invoke('openai','x',request)
    assert len(sent)==3 and [x['attempt'] for x in c.records]==[1,2,3]
    assert [x['status'] for x in read(settings.ledger)['calls']]==['failed']*3


@pytest.mark.parametrize('limiter',['money','call_count','deadline'])
def test_no_retry_sent_when_remaining_budget_or_deadline_is_insufficient(tmp_path,monkeypatch,limiter):
    settings,c=selected(tmp_path,max_usd=.003 if limiter=='money' else 1,max_calls=1 if limiter=='call_count' else 10)
    request,sent=wire([503])
    if limiter=='deadline':
        remaining=iter([60,60,0]) # first request allowed, backoff allowed, next request denied
        monkeypatch.setattr(c.budget,'remaining_s',lambda:next(remaining))
    with pytest.raises(BudgetExceeded):c.invoke('openai','x',request)
    assert len(sent)==1 and len(read(settings.ledger)['calls'])==1


def test_initial_deadline_blocks_first_request(tmp_path,monkeypatch):
    settings,c=selected(tmp_path);request,sent=wire([])
    monkeypatch.setattr(c.budget,'remaining_s',lambda:0)
    with pytest.raises(BudgetExceeded):c.invoke('openai','x',request)
    assert not sent and not read(settings.ledger)['calls']


def test_first_attempt_identity_and_reader_assets_unchanged(tmp_path):
    settings,_=selected(tmp_path)
    old=replace(settings,reader_profile=ROOT/'profiles/submission/sol_p1.json')
    v1=read(old.reader_profile);v2=read(settings.reader_profile)
    assert {k for k in v1 if v1[k]!=v2[k]}=={'version','max_attempts'}
    fixture=read(ROOT/'examples/acceptance/source_fidelity_v2/faithful_paraphrase.pack.json')
    pack=[Hit.model_validate(h) for h in fixture['pack']]
    first=request_parts('За какой срок осматривают оборудование?',pack,settings.reader().prompt(),ROOT)
    previous=request_parts('За какой срок осматривают оборудование?',pack,old.reader().prompt(),ROOT)
    assert first[1:]==previous[1:]
    def identity(s,parts):
        b=s.profile().generators['openai']
        result=model_request(*parts[:3],binding=b,endpoint='https://api.sosana.art/api',max_tokens=b.max_output_tokens)
        result['messages']=parts[5]
        return result
    assert identity(settings,first)==identity(old,previous)
    assert settings.pipeline()==old.pipeline()
