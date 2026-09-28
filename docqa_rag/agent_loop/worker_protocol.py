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
