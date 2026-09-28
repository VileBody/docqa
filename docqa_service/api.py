import secrets
from typing import Literal
from fastapi import FastAPI,Depends,Header,HTTPException,UploadFile,File,Form
from pydantic import BaseModel,ConfigDict,Field,field_validator
from docqa_rag.util import BudgetExceeded,ModelUnavailable,ModelOutputInvalid,InvalidEvidence
from .config import Settings
from .core import Service,Missing,NotReady
from .registry import Busy

class Question(BaseModel):
    model_config=ConfigDict(extra='forbid')
    document_id:str=Field(pattern=r'^[0-9a-f]{64}$')
    question:str=Field(min_length=1,max_length=12000)
    generator:Literal['qwen','openai']|None=None
    diagnostics:bool=False
    query_policy:Literal['raw','anchor','paraphrase','structured','query2doc','hyde','multiview','decomposition']='raw'
    planner_family:Literal['qwen','openai']='qwen'
    @field_validator('question')
    @classmethod
    def nonblank(cls,value):
        if not value.strip():raise ValueError('Question cannot be blank')
        return value

class DocumentRecord(BaseModel):
    doc_id:str;task_id:str;title:str;source_sha256:str;source_format:str
    classification:str;pipeline:str;status:Literal['queued','processing','ready','failed']
    created_at:float;error:str|None=None;attempts:int
    generation:str|None=None

class Accepted(BaseModel):
    task_id:str;document:DocumentRecord;dispatch:str

class SourceCitation(BaseModel):
    source_id:str;document_id:str;page:int;text:str;char_start:int;char_end:int
    text_sha256:str;source_sha256:str;source_format:Literal['txt','pdf']
    boxes:list[dict];match_policy:str

class AnswerOut(BaseModel):
    answer:str;found:bool;citations:list[SourceCitation];status:str;support_check:str
    request_id:str;generation:str;fingerprint:str;backend:str;shadow:dict
    diagnostics:dict|None=None

def create_app(settings=None,dispatcher=None):
    settings=settings or Settings.environment()
    if len(settings.api_key)<16:raise ValueError('Set DOCQA_API_KEY to at least 16 characters')
    service=Service(settings);app=FastAPI(title='DocQA TXT-first',version='0.3.0')
    app.state.service=service
    def auth(authorization:str=Header(default='')):
        if not secrets.compare_digest(authorization,'Bearer '+settings.api_key):raise HTTPException(401,'Invalid API credential')
    @app.get('/health')
    def health():return {'status':'up','profile':'txt-first','models_enabled':settings.model_calls,'test_mode':settings.test_mode}
    @app.post('/documents',status_code=202,response_model=Accepted,dependencies=[Depends(auth)])
    def upload(file:UploadFile=File(...),classification:Literal['synthetic','public']=Form(...)):
        data=file.file.read(30*1024*1024+1)
        try:record=service.upload(data,file.filename or '',classification)
        except ValueError as exc:raise HTTPException(422,str(exc)) from None
        from .tasks import dispatch_one
        sent=(dispatcher or dispatch_one)(service.registry,record['task_id']) if record['status']=='queued' else False
        return {'task_id':record['task_id'],'document':record,'dispatch':'sent' if sent else 'durable_registry'}
    @app.get('/tasks/{task_id}',response_model=DocumentRecord,dependencies=[Depends(auth)])
    def task(task_id:str):
        try:return service.task(task_id)
        except Missing:raise HTTPException(404,'Unknown task') from None
    @app.get('/documents',response_model=list[DocumentRecord],dependencies=[Depends(auth)])
    def documents():return service.list_documents()
    @app.post('/tasks/{task_id}/retry',status_code=202,response_model=DocumentRecord,dependencies=[Depends(auth)])
    def retry(task_id:str):
        try:
            record=service.retry(task_id)
            from .tasks import dispatch_one
            if record['status']=='queued':(dispatcher or dispatch_one)(service.registry,task_id)
            return record
        except Missing:raise HTTPException(404,'Unknown task') from None
        except Busy:raise HTTPException(409,'Worker publication lease active') from None
    @app.post('/questions',response_model=AnswerOut,dependencies=[Depends(auth)])
    def question(q:Question):
        selected=settings.reader() if getattr(settings,'reader_profile',None) is not None else None
        if selected and (q.generator not in {None,selected.generator} or q.query_policy!=selected.query_policy):
            raise HTTPException(422,'Request differs from selected submission profile')
        try:return service.answer(q.document_id,q.question,q.generator,q.diagnostics,q.query_policy,q.planner_family)
        except Missing:raise HTTPException(404,'Unknown document') from None
        except NotReady:raise HTTPException(409,'Document/pipeline not ready') from None
        except BudgetExceeded:raise HTTPException(429,'Authorized model budget exhausted') from None
        except ModelOutputInvalid:raise HTTPException(502,'Model output invalid; not a refusal or transient availability failure') from None
        except (ModelUnavailable,InvalidEvidence):raise HTTPException(503,'Model or evidence validation failed; not a not_found answer') from None
        except Exception:raise HTTPException(503,'Published index unavailable or invalid') from None
    return app
