"""Opt-in frozen reader experiment. No retrieval, corpus import or prompt search."""
import json
from pathlib import Path
from pydantic import Field
from .types import Contract,SpanDraft,Draft
from .adapters import ChatAdapter,generation_request
from .contexts import messages_and_schema,measure_messages
from .evidence import EvidenceIssue
from .util import read,digest

class ReaderPrompt(Contract):
    version:str
    prompt: str = Field(pattern=r'^P[01]$')
    system_file:str|None=None
    system_sha256:str|None=None
    examples_file:str|None=None
    examples_sha256:str|None=None
    public_policy:str='strict_answerability_v1'


def request_parts(question,pack,config,root):
    schema,system,payload,catalog=generation_request(question,pack,'span')
    examples=[]
    if config.prompt=='P1':
        path=Path(root)/config.system_file
        if digest(path.read_bytes())!=config.system_sha256:raise ValueError('Frozen candidate system changed')
        system=path.read_text();path=Path(root)/config.examples_file
        if digest(path.read_bytes())!=config.examples_sha256:raise ValueError('Frozen teaching examples changed')
        rows=read(path)
        from .types import StrictSpanDraft
        for row in rows:
            if row.get('purpose')!='teaching_only_not_benchmark':raise ValueError('Only fixed teaching demonstrations allowed')
            StrictSpanDraft.model_validate(row['output'])
            examples += [{'role':'user','content':json.dumps(row['input'],ensure_ascii=False)},
                         {'role':'assistant','content':json.dumps(row['output'],ensure_ascii=False)}]
        payload={**payload,'contract_version':config.version}
    messages,structure=messages_and_schema(schema,system,payload)
    messages=[messages[0],*examples,messages[1]]
    return schema,system,payload,catalog,examples,messages,structure


class ReaderAdapter(ChatAdapter):
    def __init__(self,*args,prompt_config,workspace_root,**kwargs):
        super().__init__(*args,reference_mode='span',enforce_context_limits=True,**kwargs)
        self.prompt_config=prompt_config;self.workspace_root=Path(workspace_root)
    def generate(self,question,hits):
        schema,system,payload,catalog,examples,messages,structure=request_parts(question,hits,self.prompt_config,self.workspace_root)
        wire=self.structured(schema,system,payload,self.binding.family,demonstrations=examples)
        self.last_observation.update(issued_catalog=catalog,wire_draft=wire.model_dump(),
            reader_prompt=self.prompt_config.model_dump(),full_messages_sha256=digest(messages),
            schema_sha256=digest(structure),prior_user_history=[],teaching_pairs=len(examples)//2)
        # Keep raw output unchanged. Strict public-state validation happens in the
        # projection/renderer, so invalid outputs remain in the fixed denominator.
        draft=Draft.model_validate(wire.model_dump())
        for claim in draft.claims:
            for ref in claim.references:
                if ref.source_id not in catalog:raise EvidenceIssue('unknown_source_id','Unissued source alias')
                ref.source_id=catalog[ref.source_id]['source_id']
        return draft


def full_context(binding,question,pack,config,root):
    _,_,_,_,_,messages,structure=request_parts(question,pack,config,root)
    return measure_messages(binding,messages,structure,binding.max_output_tokens)


def support_review_input(question,claim,quotes,user_operands=()):
    """Optional shadow review receives explicitly attributed user operands.

    No critic call is made here; the old claim/quotes-only checker cannot be the
    sole arithmetic judge. Operands are reviewer annotations, not source facts.
    """
    return {'original_question':question,'claim':claim,'quotes':quotes,
            'user_operands':[{'value':x,'provenance':'user_question_not_source'} for x in user_operands]}
