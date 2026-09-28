from dataclasses import replace
from pathlib import Path
import httpx,pytest,os
from docqa_service.config import Settings,ROOT
from docqa_service.submission import Submission
from docqa_service.models import calls
from docqa_rag.budget import Budget
from docqa_rag.adapters import Calls
from docqa_rag.util import ModelUnavailable

@pytest.fixture(autouse=True)
def isolated_settings(monkeypatch):
 for key in list(os.environ):
  if key.startswith('DOCQA_'):monkeypatch.delenv(key,raising=False)

def settings(tmp_path):
 return Settings(tmp_path,'','','',ROOT/'profiles/submission/sol_p1_runtime.json',reader_profile=ROOT/'profiles/submission/sol_p1.json')

def test_selected_wire_and_pipeline(tmp_path,monkeypatch):
 monkeypatch.setenv('DOCQA_OPENAI_BASE_URL','https://api.sosana.art/api');monkeypatch.setenv('DOCQA_OPENAI_API_KEY','test-key')
 s=settings(tmp_path);p=s.profile();selected=s.reader();a=selected.adapter(p,None)
 assert a.binding.model_id=='gpt-6-sol' and a.request_timeout_s==180
 assert a.client.temperature is None and a.client.reasoning_effort=='medium'
 assert s.pipeline()==replace(s,profile_path=ROOT/'profiles/stage_2/txt.json',reader_profile=ROOT/'profiles/submission/luna_p1.json').pipeline()
 monkeypatch.setenv('DOCQA_OPENAI_OUTPUT_TOKENS','2048')
 with pytest.raises(ValueError,match='binding differs'):s.profile()

def test_selected_prices_and_one_attempt(tmp_path):
 ledger=tmp_path/'ledger.json';Budget(ledger,1,10,600)
 s=replace(settings(tmp_path),model_calls=True,ledger=str(ledger));c=calls(s)
 assert c.prices['openai']['input_per_million']==.6 and c.max_attempts==1
 n=[]
 def fail():n.append(1);raise httpx.ReadTimeout('test')
 with pytest.raises(ModelUnavailable):c.invoke('openai','x',fail)
 assert len(n)==1
 assert Submission.load(ROOT/'profiles/submission/luna_p1.json').max_attempts==3
