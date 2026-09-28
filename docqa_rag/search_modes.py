"""Initial stage3 retrieval modes over the same bounded source tool.

This is structured-JSON tool dispatch, not a claim of native tool-calling support.
Reactive/research remain a later, separately measured extension.
"""
from pydantic import Field
from .types import Contract,Hit
from .search_evidence import SearchRequest,InvalidToolRequest
from .planning import STRUCTURED_SYSTEM,DECOMPOSITION_SYSTEM
from .selection import select_pack
from .contexts import fit_common_pack
from .util import BudgetExceeded,ModelOutputInvalid,digest

MODE_VERSION='bounded-initial-modes-v1'

class SearchBatch(Contract):
    requests:list[SearchRequest]=Field(min_length=1,max_length=2)

def retrieve_mode(tool,mode='flow',planner=None):
    if mode not in {'flow','tool_once','static_batch'}:raise ValueError('Mode not implemented')
    trace={'mode':mode,'version':MODE_VERSION,'dispatch':'structured_json_not_native_tools',
           'controller_calls':0,'critic_mode':'shadow','original_question':tool.question}
    if mode=='flow':requests=[SearchRequest(query=tool.question)]
    else:
        if planner is None:raise ValueError('Explicit controller adapter required')
        trace['controller_calls']=1
        system=(STRUCTURED_SYSTEM if mode=='tool_once' else DECOMPOSITION_SYSTEM).split('\nНейтральный пример:')[0]
        # Reuse task semantics, but make the actual tool request schema explicit.
        system=system.replace('в views','в query каждого requests' if mode=='static_batch' else 'в query')
        system+=' Верни объект по приложенной tool schema. Поле source_evidence_ids=[]: выдачи ещё не было. Не выбирай документ, snapshot, лимиты или права. '
        system+=('Сформируй один query; views не является полем этой схемы.' if mode=='tool_once' else 'Сформируй до двух requests заранее. Выдачу первого запроса для изменения второго использовать нельзя; views не является полем этой схемы.')
        system+=('\nНейтральный пример: «Когда возвращают залог?» → {"query":"Условия и момент возврата залога","source_evidence_ids":[]}.' if mode=='tool_once' else '\nНейтральный пример: «Когда возвращают залог и кто его перечисляет?» → {"requests":[{"query":"Условия и момент возврата залога","source_evidence_ids":[]},{"query":"Лицо, ответственное за перечисление залога","source_evidence_ids":[]} ]}.')
        schema=SearchRequest if mode=='tool_once' else SearchBatch
        proposed=planner.structured(schema,system,{'original_question':tool.question},planner.binding.family)
        trace['controller_observation']=getattr(planner,'last_observation',{})
        requests=[proposed] if mode=='tool_once' else proposed.requests
    trace['frozen_requests']=[r.model_dump() for r in requests]
    # Reject the entire static batch before any retrieval on invalid anchors/scope.
    from .search_evidence import anchor_provenance
    for request in requests:
        if request.source_evidence_ids:raise InvalidToolRequest('Initial batch cannot reference unseen sources')
        anchor_provenance(tool.question,request.query,{})
    for request in requests:
        tool.search(request)
        if tool.stopped:break
    if tool.stopped in {'timeout','model_unavailable','invalid_tool_result','invalid_tool_request','budget_exhausted'}:
        return [],{**trace,'status':tool.stopped,'searches':tool.history,'actual':dict(tool.actual)}
    # Rerank the union using the ORIGINAL question in every mode. Per-query
    # relevance scores are not silently mixed as if they were comparable.
    candidates=[Hit(chunk=r['fragment'],retrieval_score=r['rerank_score']) for r in tool.receipts.values()]
    if tool.actual['rerank_pairs']+len(candidates)>tool.limits.rerank_pairs:
        tool.stopped='budget_exhausted'
        return [],{**trace,'status':tool.stopped,'searches':tool.history,'actual':dict(tool.actual)}
    tool._time()
    if tool.index.fingerprint!=tool.snapshot:raise InvalidToolRequest('Snapshot changed before final pack')
    tool.actual['rerank_pairs']+=len(candidates)
    ranked=tool.reranker.rank(tool.question,candidates) if candidates else []
    tool._time()
    if tool.index.fingerprint!=tool.snapshot:raise InvalidToolRequest('Snapshot changed during final rerank')
    by_source={h.chunk.source_id:h.chunk.model_dump() for h in candidates}
    if len(ranked)!=len(candidates) or len({h.chunk.source_id for h in ranked})!=len(ranked) or any(by_source.get(h.chunk.source_id)!=h.chunk.model_dump() for h in ranked):
        raise InvalidToolRequest('Final reranker altered source union')
    pack,selection=select_pack(candidates,ranked,tool.profile)
    pack,context=fit_common_pack(tool.question,pack,tool.profile)
    trace.update(status='success',stop_reason=tool.stopped or 'planned_searches_complete',searches=tool.history,
                 actual=dict(tool.actual),selection=selection,common_context=context,
                 pack_hash=digest([h.chunk.model_dump() for h in pack]))
    return pack,trace
