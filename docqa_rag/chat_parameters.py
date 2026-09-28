"""Versioned Chat Completions parameters; never silently downgrade reasoning."""
VERSION = 'reasoning-chat-parameters-v1'
LEVELS = {'low', 'medium', 'high', 'xhigh', 'max'}

def reasoning_effort(binding):
    if binding.family != 'openai' or not binding.model_id.startswith(('gpt-6', 'gpt-5')):
        return None
    effort = 'none' if binding.reasoning == 'disabled' else binding.reasoning
    if binding.model_id.startswith('gpt-6-astra') and effort not in LEVELS:
        raise ValueError('Astra requires an explicit supported reasoning effort')
    if binding.model_id.startswith(('gpt-6-sol', 'gpt-6-luna')) and effort not in LEVELS | {'none'}:
        raise ValueError('Unsupported GPT-6 reasoning effort')
    return effort

def chat_parameters(binding, max_tokens):
    effort = reasoning_effort(binding)
    if effort is not None and effort != 'none':
        return {'reasoning_effort': effort, 'max_completion_tokens': max_tokens}
    return {'temperature': 0, 'max_tokens': max_tokens,
            **({'reasoning_effort': effort} if effort is not None else {})}
