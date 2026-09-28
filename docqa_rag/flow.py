"""Stateless flow. Only backend-validated source receipts reach public citations."""
import re,time
from .types import QueryPlan, Citation, Answer, Verdict
from .util import InvalidEvidence, ModelOutputInvalid, digest
from .tokenization import count as token_count, binding as tokenizer_binding


def plan_query(question,policy='raw'):
    if not question.strip() or len(question)>12000: raise ValueError('Question empty/too long')
    anchors=re.findall(r'(?<!\w)(?:№\s*)?(?=[\w/.,-]*\d)\w+(?:[-/.,]\w+)*(?!\w)',question)
    anchors=list(dict.fromkeys(anchors))
    # Original spelling/case/digits remain byte-for-byte. Dense optional anchors are additive only.
    return QueryPlan(raw=question,lexical=question,dense=question if policy=='raw' else question+'\n'+' '.join(anchors),anchors=anchors,policy=policy)

class ExtractiveCritic:
    """Conservative operational check for extractive claims, not semantic completeness."""
    def check(self,claim,quotes):
        supported=any(claim in q for q in quotes)
        return Verdict(supported=supported,reason='verbatim claim supported' if supported else 'non-extractive claim needs semantic critic')


from .evidence import resolve_claim, issued_sources
from .selection import select_pack

class Flow:
    def __init__(self,index,embedder,reranker,generator,profile,critic=None,critic_mode='enforce',*,answerability_policy='legacy',answer_fit=None,answer_fit_mode='shadow'):
        if critic_mode not in {'enforce','shadow','off'}:raise ValueError('Invalid critic mode')
        if answerability_policy not in {'legacy','strict_answerability_v1'}:raise ValueError('Invalid answerability policy')
        self.answerability_policy=answerability_policy
        if answer_fit_mode not in {'shadow','enforce'}:raise ValueError('Invalid answer-fit mode')
        self.answer_fit=answer_fit; self.answer_fit_mode=answer_fit_mode
        self.critic_mode=critic_mode
        self.index=index; self.embedder=embedder; self.reranker=reranker; self.generator=generator; self.profile=profile; self.critic=critic; self.debug=None
    def retrieve(self,doc_id,question,branch='hybrid',use_reranker=True):
        t=time.perf_counter(); plan=plan_query(question,self.profile.query_policy)
        candidates=self.index.search(doc_id,plan,self.embedder,self.profile.candidate_limit,branch,self.profile.fusion)
        retrieval_s=time.perf_counter()-t; t=time.perf_counter()
        ranked=self.reranker.rank(question,candidates) if use_reranker else candidates
        packed,selection=select_pack(candidates,ranked,self.profile,use_reranker)
        trace={'query_plan':plan.model_dump(),'snapshot':self.index.fingerprint,'dataset_hash':self.index.spec['dataset_hash'],
            'branch':branch,'reranker':use_reranker,'candidates':[h.model_dump() for h in candidates],
            'reranked':[h.model_dump() for h in ranked],'pack':[h.model_dump() for h in packed],
            'reference_tokenizer':tokenizer_binding(),**selection,
            'source_aliases':{f'S{i+1}':h.chunk.source_id for i,h in enumerate(packed)},
            'pack_hash':digest([h.chunk.model_dump() for h in packed]),'retrieval_s':retrieval_s,'rerank_pack_s':time.perf_counter()-t}
        return packed,trace
    def answer_pack(self,doc_id,question,pack):
        if doc_id not in self.index.documents: raise InvalidEvidence('Unknown document')
        for h in pack:
            c=h.chunk; doc=self.index.documents[doc_id]
            if c.doc_id!=doc_id or doc.text[c.char_start:c.char_end]!=c.text: raise InvalidEvidence('Invalid frozen pack')
        if not pack:
            return Answer(answer='Недостаточно подтверждённых сведений в найденных источниках.',found=False,citations=[],status='not_found',support_check='no_sources'),{}
        if self.profile.debug_artifacts and self.debug is None:
            raise ValueError('Debug-enabled generation requires a protected DebugReceipt sink before inference')
        from .reader_control import ReaderAdapter
        if self.profile.enforce_context_limits and not isinstance(self.generator,ReaderAdapter):
            from .adapters import generation_request
            from .contexts import measure
            schema,system,payload,_=generation_request(question,pack,self.profile.reference_mode)
            from .contexts import ContextLimit
            try:boundaries=[measure(b,schema,system,payload) for b in self.profile.generators.values()]
            except ContextLimit as exc:
                if self.debug:self.debug.capture('context_rejected',receipt=exc.receipt,transmitted=False)
                raise
            if self.debug:self.debug.capture('context_preflight',boundaries=boundaries)
        try:
            draft=self.generator.generate(question,pack)
        except Exception as exc:
            if self.debug:self.debug.capture('generation_failed',observation=getattr(self.generator,'last_observation',{}),issue=getattr(exc,'code',type(exc).__name__))
            raise
        return self.validate_draft(doc_id,pack,draft,question=question)

    def validate_draft(self,doc_id,pack,draft,*,question=None):
        if self.debug:
            self.debug.capture('before_validation',draft=draft.model_dump(),issued_catalog=issued_sources(pack,self.profile.reference_mode)[1],
                pack_hash=digest([h.chunk.model_dump() for h in pack]),observation=getattr(self.generator,'last_observation',{}))
        try:return self._validate_draft(doc_id,pack,draft,question=question)
        except Exception as exc:
            if self.debug:self.debug.capture('validation_failed',issue=getattr(exc,'code',type(exc).__name__),message=str(exc)[:500],critic_observation=getattr(self.critic,'last_observation',{}))
            raise

    def _validate_draft(self,doc_id,pack,draft,*,question=None):
        if self.answerability_policy=='strict_answerability_v1':
            from .types import StrictDraft
            from pydantic import ValidationError
            try:draft=StrictDraft.model_validate(draft.model_dump())
            except ValidationError as exc:
                raise ModelOutputInvalid('Inconsistent answerability status/claims/missing') from exc
        if draft.status in {'not_found','ambiguous','contradictory'}:
            if draft.status=='not_found' and draft.claims: raise InvalidEvidence('not_found with claims')
            return Answer(answer={'not_found':'Недостаточно сведений в источниках.','ambiguous':'Источники неоднозначны; однозначный ответ не подтверждён.',
                'contradictory':'В источниках обнаружено противоречие; однозначный ответ не подтверждён.'}[draft.status],found=False,citations=[],status=draft.status,support_check='abstained'),{'draft':draft.model_dump()}
        if not draft.claims: raise InvalidEvidence('Positive draft without claims')
        citations=[]; checks=[]; resolved_claims=[]
        for claim in draft.claims:
            resolved=resolve_claim(claim,pack,self.index.documents,doc_id)
            resolved_claims.append(resolved)
            if self.critic and self.critic_mode!='off':
                try:
                    verdict=self.critic.check(claim.text,[c.text for c in resolved]); checks.append(verdict.model_dump())
                except Exception as exc:
                    if self.critic_mode!='shadow':raise
                    checks.append({'status':'invalid_output' if isinstance(exc,ModelOutputInvalid) else 'unavailable','error_type':type(exc).__name__})
                    verdict=None
                if verdict and not verdict.supported and self.critic_mode=='enforce':
                    return Answer(answer='Поддержка ответа источниками не подтверждена.',found=False,citations=[],status='unsupported',support_check='critic_rejected'),{'draft':draft.model_dump(),'checks':checks}
            citations.extend(resolved)
        fit_details={}
        if self.answer_fit is not None:
            from .answer_fit import check_input,validate_report,project
            try:
                payload=check_input(question,draft,pack,resolved_claims)
                report=validate_report(self.answer_fit.check(question,draft,pack,resolved_claims),payload)
                fit_details={'mode':self.answer_fit_mode,'report':report.model_dump()}
                guarded=project(report,draft.status)
            except Exception as exc:
                fit_details={'mode':self.answer_fit_mode,'technical_error':type(exc).__name__}
                if self.debug:self.debug.capture('answer_fit_failed',**fit_details,observation=getattr(self.answer_fit,'last_observation',{}))
                if self.answer_fit_mode=='enforce':raise
                guarded=None
            if self.debug:self.debug.capture('answer_fit',**fit_details,observation=getattr(self.answer_fit,'last_observation',{}))
            if guarded is not None and self.answer_fit_mode=='enforce':
                return guarded,{'draft':draft.model_dump(),'checks':checks,'answer_fit':fit_details}
        answer='\n'.join(c.text for c in draft.claims)
        if draft.status=='partial':
            answer='Частичный ответ:\n'+answer+'\nОстальные запрошенные сведения не подтверждены.'
            if draft.missing and self.answerability_policy=='legacy':
                answer+='\nНе подтверждено источниками: '+'; '.join(draft.missing)
        return Answer(answer=answer,found=True,citations=citations,status=draft.status,
            support_check=('citation_authenticity_shadow' if self.critic_mode=='shadow' else 'critic_passed') if self.critic and self.critic_mode!='off' else 'citation_authenticity_only'),{'draft':draft.model_dump(),'checks':checks,'critic_mode':self.critic_mode,**({'answer_fit':fit_details} if fit_details else {})}
    def ask(self,doc_id,question,branch='hybrid',use_reranker=True):
        t=time.perf_counter(); pack,trace=self.retrieve(doc_id,question,branch,use_reranker)
        answer,details=self.answer_pack(doc_id,question,pack)
        trace.update(details); trace['answer']=answer.model_dump(); trace['total_s']=time.perf_counter()-t
        return answer,trace
