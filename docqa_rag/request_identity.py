"""Safe logical model request identity, shared by runtime and offline evaluation."""
import importlib.metadata
from langchain_openai.chat_models.base import _convert_to_openai_response_format
from .contexts import messages_and_schema

def model_request(schema, system, payload, *, binding, endpoint, max_tokens=512):
    messages, structure = messages_and_schema(schema, system, payload)
    from .chat_parameters import chat_parameters,VERSION
    options=chat_parameters(binding,max_tokens)
    result = {
        'messages': messages,
        'response_format': _convert_to_openai_response_format(structure, strict=True),
        'model': binding.model_id, 'revision': binding.revision,
        'endpoint': endpoint.rstrip('/'), 'provider': binding.provider,
        **options,
        'history': [], 'stream':False,
        'sdk_versions':{n:importlib.metadata.version(n) for n in ('langchain-openai','langchain-core','pydantic','openai')},
        'contract': 'citation-evaluator-request-v1',
    }
    # Preserve old logical request hashes for non-reasoning/Qwen bindings.
    if 'reasoning_effort' not in result:result['reasoning_effort']=None
    if 'max_completion_tokens' in options:result['parameter_contract']=VERSION
    return result
