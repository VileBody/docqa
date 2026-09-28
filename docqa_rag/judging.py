"""One primary judgment per complete logical request; instability is separate."""
import fcntl
import importlib.metadata
from pathlib import Path
from langchain_openai.chat_models.base import _convert_to_openai_response_format
from .contexts import messages_and_schema
from .util import digest, read, write
from .response_contract import response_contract

SEMANTIC_RUBRIC_V2 = '''Judge semantic quality separately from the deterministic response/API contract and benchmark abstention polarity. Judge support using only cited source evidence, not reference/gold evidence or uncited context. Judge completeness against required reference facts and AND/OR evidence alternatives. A critical error is a materially false or unsupported main answer, reversal of a condition/negation, wrong role, number, unit, period or document scope that changes the answer. A missing required fact makes the answer incomplete; it is not automatically a critical error unless the omission makes the asserted rule false. A partial response can be useful without satisfying strict benchmark abstention. Do not repair or supplement the actual answer using the gold.'''

def response_evaluator_request(schema,system,payload,*,answer,service_status,binding,endpoint,max_tokens=512):
    """Offline-only complete identity for future v2 judgments. No call is made."""
    full_payload={**payload,'actual_answer':answer,'actual_found':answer.get('found') if answer else None,
                  'actual_status':{'service':service_status,'answer':answer.get('status') if answer else None,
                                   'support_check':answer.get('support_check') if answer else None},
                  'response_contract':response_contract(answer,service_status)}
    request=evaluator_request(schema,system+'\n'+SEMANTIC_RUBRIC_V2,full_payload,binding=binding,endpoint=endpoint,max_tokens=max_tokens)
    request['contract']='response-evaluator-request-v2'
    request['semantic_rubric']='semantic-quality-v2'
    return request


from .request_identity import model_request as evaluator_request


class PrimaryJudgments:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def get_or_evaluate(self, request, evaluate):
        key = digest(request)
        path = self.directory / (key + '.json')
        with (self.directory / (key + '.lock')).open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if path.exists():
                record = read(path)
                if record['request'] != request or record['input_id'] != key:
                    raise ValueError('Evaluator cache integrity mismatch')
                return record, True
            judgment = evaluate()
            if hasattr(judgment, 'model_dump'):
                judgment = judgment.model_dump()
            record = {'input_id': key, 'request': request, 'judgment': judgment,
                      'role': 'primary', 'independent_sample': False}
            write(path, record)
            return record, False

    def record_instability(self, request, judgment, experiment_id, replicate_id):
        # Never overwrite or select a better primary judgment.
        key = digest([request, experiment_id, replicate_id])
        path = self.directory / 'instability' / (key + '.json')
        if path.exists():
            raise FileExistsError(path)
        write(path, {'input_id': digest(request), 'request': request,
                     'experiment_id': experiment_id, 'replicate_id': replicate_id,
                     'judgment': judgment, 'role': 'instability_only'})
