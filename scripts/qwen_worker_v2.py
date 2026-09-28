#!/usr/bin/env python3
"""Ephemeral authenticated model worker; launch via stage1_session.py only.

Watchdog starts BEFORE dependency installation/model loading and survives loss of
local client. No corpus, gold, evaluator or account inventory is uploaded.
"""
import json, os, subprocess, sys, threading, time, urllib.request, urllib.error
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

"""Pure, CPU-testable rendering used verbatim by the ephemeral v2 worker."""
import hashlib
import json
import re

RERANK_CONTRACT='explicit-instruction-v2'
DEFAULT_RERANK_INSTRUCTION='Given a question, retrieve passages that answer it.'

def render_rerank(query,document,instruction=DEFAULT_RERANK_INSTRUCTION):
    if not isinstance(instruction,str) or not instruction.strip():raise ValueError('Explicit nonempty rerank instruction required')
    return ('<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
            +'<Instruct>: '+instruction+'\n<Query>: '+query+'\n<Document>: '+document
            +'<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n')

def validate_tool_history(messages):
    pending=set();seen=set()
    for m in messages:
        if m['role']=='tool':
            if m.get('tool_call_id') not in pending:raise ValueError('Unmatched tool response')
            pending.remove(m['tool_call_id'])
        else:
            if pending:raise ValueError('Tool result missing before next message')
            for call in m.get('tool_calls',[]):
                if m['role']!='assistant' or not call.get('id') or call['id'] in seen:raise ValueError('Invalid tool call ID')
                pending.add(call['id']);seen.add(call['id'])
    if pending:raise ValueError('Unanswered tool calls')

def render_chat(tokenizer,messages,tools=None):
    validate_tool_history(messages)
    normalized=[]
    for message in messages:
        m=dict(message)
        if m.get('tool_calls'):
            m['tool_calls']=[]
            for call in message['tool_calls']:
                c={**call,'function':dict(call['function'])}
                if isinstance(c['function']['arguments'],str):
                    try:c['function']['arguments']=json.loads(c['function']['arguments'])
                    except ValueError:pass  # Preserve malformed public call text for the bounded repair turn.
                m['tool_calls'].append(c)
        normalized.append(m)
    return tokenizer.apply_chat_template(normalized,tools=tools or None,tokenize=False,add_generation_prompt=True)

def parse_native(content,tools,request_id):
    allowed={t['function']['name'] for t in tools};calls=[]
    segments=re.findall(r'<tool_call>\s*(.*?)\s*</tool_call>',content,re.S)
    if content.count('<tool_call>')!=len(segments):raise ValueError('Truncated native tool call')
    for i,segment in enumerate(segments):
        call=json.loads(segment)
        if set(call)!={'name','arguments'} or call['name'] not in allowed or not isinstance(call['arguments'],dict):raise ValueError('Invalid native tool call')
        calls.append({'id':'call_'+request_id+'_'+str(i),'type':'function','function':{'name':call['name'],'arguments':json.dumps(call['arguments'],ensure_ascii=False)}})
    return {'role':'assistant','content':re.sub(r'<tool_call>.*?</tool_call>','',content,flags=re.S).strip() or None,**({'tool_calls':calls} if calls else {})}

DEADLINE=float(os.environ['DOCQA_DEADLINE'])
IDLE=float(os.environ.get('DOCQA_IDLE_SECONDS','240'))
TOKEN=os.environ['DOCQA_WORKER_TOKEN']
LAST_HEARTBEAT=time.time(); LAST_WORK=time.time(); READY=False; HARDWARE={}
RUNPOD_KEY=os.environ['RUNPOD_API_KEY']; POD_ID=os.environ['RUNPOD_POD_ID']
SESSION=os.environ['DOCQA_SESSION_ID']

def terminate_self(reason):
    url='https://rest.runpod.io/v1/pods/'+POD_ID
    for attempt in range(3):
        try:
            req=urllib.request.Request(url,method='DELETE',headers={'Authorization':'Bearer '+RUNPOD_KEY})
            with urllib.request.urlopen(req,timeout=15) as response: response.read()
            print(json.dumps({'watchdog':'delete_requested','reason':reason,'pod_id':POD_ID}),flush=True)
            return
        except urllib.error.HTTPError as e:
            if e.code==404: return
        except Exception: pass
        time.sleep(2**attempt)
    print(json.dumps({'watchdog':'delete_failed_manual_action_required','pod_id':POD_ID}),flush=True)
    # API outage cannot guarantee billing stops. Keep retrying without model compute.

def watchdog():
    while True:
        time.sleep(5)
        reason=None
        if time.time()>=DEADLINE: reason='deadline'
        elif READY and time.time()-LAST_WORK>IDLE: reason='idle'
        elif time.time()-LAST_HEARTBEAT>600: reason='lost_client'
        if reason:
            terminate_self(reason); os._exit(3)
threading.Thread(target=watchdog,daemon=True).start()

class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args): pass
    def reply(self,code,data):
        raw=json.dumps(data).encode(); self.send_response(code); self.send_header('Content-Type','application/json'); self.send_header('Content-Length',str(len(raw))); self.end_headers(); self.wfile.write(raw)
    def authorized(self): return self.headers.get('Authorization')=='Bearer '+TOKEN
    def do_GET(self):
        if not self.authorized(): return self.reply(401,{'error':'unauthorized'})
        if self.path=='/health': return self.reply(200,{'ready':READY,'session_id':SESSION,'pod_id':POD_ID,'deadline':DEADLINE,'models':BINDINGS,'hardware':HARDWARE,'structured_output_capability':'schema_instruction_with_post_validation','constrained_decoding':False,'native_tools':'qwen_template_xml_calls_v2','rerank_instruction_contract':RERANK_CONTRACT,'chat_template_sha256':hashlib.sha256(TOKENIZERS['generator'].chat_template.encode()).hexdigest() if READY else None})
        self.reply(404,{'error':'unknown route'})
    def do_POST(self):
        global LAST_HEARTBEAT,LAST_WORK
        if not self.authorized(): return self.reply(401,{'error':'unauthorized'})
        if self.path=='/heartbeat': LAST_HEARTBEAT=time.time(); return self.reply(200,{'ok':True})
        if not READY: return self.reply(503,{'error':'loading'})
        if time.time()>=DEADLINE-30: return self.reply(503,{'error':'deadline'})
        length=int(self.headers.get('Content-Length','0'))
        if not 0<length<=256000: return self.reply(413,{'error':'request limit'})
        try:
            data=json.loads(self.rfile.read(length)); LAST_WORK=time.time()
            with MODEL_LOCK:
                result=infer(self.path,data)
            LAST_WORK=time.time(); self.reply(200,result)
        except Exception as e: self.reply(400,{'error':type(e).__name__,'message':str(e).replace(TOKEN,'[REDACTED]').replace(RUNPOD_KEY,'[REDACTED]')[:1500]})

BINDINGS=json.loads(os.environ['DOCQA_MODEL_BINDINGS']); MODEL_LOCK=threading.Lock()
server=ThreadingHTTPServer(('0.0.0.0',8000),Handler)
threading.Thread(target=server.serve_forever,daemon=True).start()
subprocess.run([sys.executable,'-m','pip','install','--disable-pip-version-check','transformers==4.51.3','accelerate==1.6.0','safetensors==0.5.3'],check=True,timeout=max(1,min(300,DEADLINE-time.time()-60)))
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
HARDWARE={'gpu_name':torch.cuda.get_device_name(0),'gpu_total_memory_bytes':torch.cuda.get_device_properties(0).total_memory,'torch_version':torch.__version__,'cuda_runtime':torch.version.cuda,'dtype':'float16'}
MODELS={}; TOKENIZERS={}
for role,b in BINDINGS.items():
    cls=AutoModel if role=='embedding' else AutoModelForCausalLM
    TOKENIZERS[role]=AutoTokenizer.from_pretrained(b['model_id'],revision=b['revision'],padding_side='left')
    MODELS[role]=cls.from_pretrained(b['model_id'],revision=b['revision'],torch_dtype=torch.float16,device_map='cuda',use_safetensors=True).eval()

def batch(role,texts,max_length=8192):
    tok=TOKENIZERS[role]; values=tok(texts,padding=True,return_tensors='pt',truncation=False)
    if values['input_ids'].shape[1]>max_length: raise ValueError('Token limit; no silent truncation')
    return {k:v.to('cuda') for k,v in values.items()}

@torch.inference_mode()
def infer(path,data):
    if path=='/v1/embeddings':
        if data['model']!=BINDINGS['embedding']['model_id']: raise ValueError('Wrong model')
        texts=data['input']; texts=[texts] if isinstance(texts,str) else texts
        if not 1<=len(texts)<=16 or not all(isinstance(t,str) for t in texts): raise ValueError('Batch size/type')
        inputs=batch('embedding',texts); hidden=MODELS['embedding'](**inputs).last_hidden_state[:,-1]
        vectors=torch.nn.functional.normalize(hidden,p=2,dim=1).float().cpu().tolist()
        return {'object':'list','model':data['model'],'data':[{'object':'embedding','index':i,'embedding':v} for i,v in enumerate(vectors)],'usage':{'prompt_tokens':int(inputs['attention_mask'].sum()),'total_tokens':int(inputs['attention_mask'].sum())}}
    if path=='/v1/rerank':
        if data['model']!=BINDINGS['reranker']['model_id'] or len(data['documents'])>200: raise ValueError('Reranker scope')
        tok=TOKENIZERS['reranker']
        instruction=data.get('instruction',DEFAULT_RERANK_INSTRUCTION)
        if data.get('instruction_contract',RERANK_CONTRACT)!=RERANK_CONTRACT:raise ValueError('Instruction contract')
        rows=[]
        for start in range(0,len(data['documents']),8):
            texts=[render_rerank(data['query'],d,instruction) for d in data['documents'][start:start+8]]
            inputs=batch('reranker',texts);logits=MODELS['reranker'](**inputs).logits[:,-1,:]
            scores=torch.softmax(logits[:,[tok.convert_tokens_to_ids('no'),tok.convert_tokens_to_ids('yes')]].float(),dim=-1)[:,1].cpu().tolist()
            rows.extend({'index':start+i,'relevance_score':s} for i,s in enumerate(scores))
        return {'results':sorted(rows,key=lambda r:-r['relevance_score']),'instruction_contract':RERANK_CONTRACT,'instruction_sha256':hashlib.sha256(instruction.encode()).hexdigest()}
    if path=='/v1/chat/completions':
        if data['model']!=BINDINGS['generator']['model_id']: raise ValueError('Wrong generator')
        tok=TOKENIZERS['generator']; messages=data['messages']
        schema=(data.get('response_format') or {}).get('json_schema',{}).get('schema')
        if schema: messages=[{'role':'system','content':'Return only valid JSON matching this schema: '+json.dumps(schema)}]+messages
        text=render_chat(tok,messages,data.get('tools'))
        inputs=batch('generator',[text],max_length=16000)
        n=int(data.get('max_tokens',data.get('max_completion_tokens',2048)))
        if not 1<=n<=2048: raise ValueError('Output token limit')
        output=MODELS['generator'].generate(**inputs,max_new_tokens=n,do_sample=False,pad_token_id=tok.eos_token_id)
        generated=output[0,inputs['input_ids'].shape[1]:]; content=tok.decode(generated,skip_special_tokens=True)
        used=inputs['input_ids'].shape[1]
        try:message=parse_native(content,data['tools'],str(time.time_ns())) if data.get('tools') else {'role':'assistant','content':content}
        except ValueError:message={'role':'assistant','content':content}
        return {'id':'qwen-'+str(time.time_ns()),'object':'chat.completion','created':int(time.time()),'model':data['model'],
            'choices':[{'index':0,'message':message,'finish_reason':'length' if len(generated)==n else 'tool_calls' if message.get('tool_calls') else 'stop'}],
            'usage':{'prompt_tokens':used,'completion_tokens':len(generated),'total_tokens':used+len(generated)}}
    raise ValueError('Unknown inference route')
READY=True; LAST_WORK=time.time()
print(json.dumps({'ready':True,'pod_id':POD_ID,'session_id':SESSION}),flush=True)
while time.time()<DEADLINE: time.sleep(1)
terminate_self('deadline_main')
