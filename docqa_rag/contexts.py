"""Full-prompt length receipts. Native Qwen vs labeled conservative provider bound."""
import json,hashlib
from functools import lru_cache
from pathlib import Path
from .util import ModelUnavailable,digest

class ContextLimit(ModelUnavailable):
    def __init__(self,receipt):super().__init__('Context limit: '+str(receipt));self.receipt=receipt

def messages_and_schema(schema,system,payload):
    structure=schema.model_json_schema()
    messages=[{'role':'system','content':system+'\nSchema:\n'+json.dumps(structure)},
              {'role':'user','content':json.dumps(payload,ensure_ascii=False)}]
    return messages,structure

@lru_cache(maxsize=4)
def native_tokenizer(path,revision,template_hash):
    from transformers import AutoTokenizer
    path=Path(path)
    if not path.is_absolute():path=Path(__file__).resolve().parents[1]/path
    provenance=json.loads((path/'provenance.json').read_text())
    if provenance['revision']!=revision:raise ValueError('Tokenizer revision mismatch')
    for name,sha in provenance['files'].items():
        if hashlib.sha256((path/name).read_bytes()).hexdigest()!=sha:raise ValueError('Tokenizer file mismatch')
    tok=AutoTokenizer.from_pretrained(path,local_files_only=True,trust_remote_code=False)
    if template_hash and hashlib.sha256(tok.chat_template.encode()).hexdigest()!=template_hash:raise ValueError('Chat template hash mismatch')
    return tok

def measure(binding,schema,system,payload,max_output=None):
    messages,structure=messages_and_schema(schema,system,payload)
    return measure_messages(binding,messages,structure,max_output)

def measure_messages(binding,messages,structure,max_output=None):
    """Count the entire transmitted conversation, including fixed teaching pairs."""
    output=max_output or binding.max_output_tokens
    from langchain_openai.chat_models.base import _convert_to_openai_response_format
    structure=_convert_to_openai_response_format(structure,strict=True)['json_schema']['schema']
    if binding.tokenizer_path:
        tok=native_tokenizer(binding.tokenizer_path,binding.tokenizer_revision,binding.chat_template_sha256)
        # Identical schema prepend and template as qwen_worker /chat/completions.
        effective=[{'role':'system','content':'Return only valid JSON matching this schema: '+json.dumps(structure)}]+messages
        rendered=tok.apply_chat_template(effective,tokenize=False,add_generation_prompt=True)
        n=len(tok.encode(rendered,add_special_tokens=True));method='native_pinned_qwen_chat_template'
    else:
        wire=json.dumps({'messages':messages,'response_format':{'type':'json_schema','json_schema':{'schema':structure,'strict':True}}},ensure_ascii=False)
        n=2*len(wire.encode())+4096;method='conservative_utf8_bound_with_envelope_not_native_tokenizer'
    receipt={'requested_alias':binding.model_id,'provider':binding.provider,'method':method,'input_count_or_bound':n,
             'model_input_budget':binding.model_input_budget,'max_output_tokens':output,'model_context_budget':binding.model_context_budget,
             'limit_source':binding.context_limit_source,'prompt_hash':digest(messages),'truncation':'none','origin_verification':binding.origin_verification}
    receipt['fits']=n<=binding.model_input_budget and output<=binding.max_output_tokens and n+output<=binding.model_context_budget
    if not receipt['fits']:raise ContextLimit(receipt)
    return receipt

def fit_common_pack(question,pack,profile):
    """Same ordered whole-chunk prefix for every generator; explicit removals.

    Apply to all compared modes before generation. Never silently use a larger
    evidence context for one model, and never change the underlying TXT/chunks.
    """
    from .adapters import generation_request
    fitted=list(pack);removed=[]
    while True:
        schema,system,payload,_=generation_request(question,fitted,profile.reference_mode)
        try:
            boundaries={family:measure(binding,schema,system,payload) for family,binding in profile.generators.items()}
            break
        except ContextLimit:
            if not fitted:raise
            removed.append(fitted.pop().chunk.source_id)
    return fitted,{'version':'shared-whole-chunk-context-v2','original_pack_hash':digest([h.chunk.model_dump() for h in pack]),
                   'final_pack_hash':digest([h.chunk.model_dump() for h in fitted]),'removed_source_ids':removed,
                   'boundaries':boundaries,'applied_identically_to_all_generators':True}
