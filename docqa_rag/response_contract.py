"""Gold-free response shape policy, shared by offline evaluation and clients.

Partial is a valid API response, not a promise of benchmark success. Gold polarity
is evaluated only in the offline projection; it never enters Flow or a tool.
"""
from .types import Answer

RESPONSE_CONTRACT_VERSION = 'answer-shape-v2'

def response_contract(answer, service_status='success'):
    issues=[]
    if service_status!='success':issues.append('service_error_not_abstention')
    try:
        parsed=Answer.model_validate(answer,strict=True)
    except (ValueError,TypeError):
        return {'version':RESPONSE_CONTRACT_VERSION,'valid':False,'issues':issues+['invalid_answer_shape']}
    if not parsed.answer.strip():issues.append('empty_message')
    if parsed.found and parsed.status not in {'complete','partial'}:issues.append('found_status_mismatch')
    if not parsed.found and parsed.status not in {'not_found','ambiguous','contradictory'}:
        issues.append('non_answer_status_not_abstention')
    for citation in parsed.citations:
        if not citation.text.strip() or citation.char_start<0 or citation.char_end<=citation.char_start or citation.page<1:
            issues.append('invalid_citation_shape')
    return {'version':RESPONSE_CONTRACT_VERSION,'valid':not issues,'issues':sorted(set(issues)),
            'citation_authenticity':'requires_source_validation_separately'}
