"""Deterministic selection with an event for every candidate; no gold access."""
from .tokenization import count

def select_pack(candidates,ranked,profile,use_reranker=True):
    origins={h.chunk.source_id:(i+1,h.retrieval_score) for i,h in enumerate(candidates)}
    packed=[];events=[];seen=set();eligible=0;chars=0;tokens=0
    for rank,h in enumerate(ranked,1):
        c=h.chunk;n=len(c.text);t=count(c.text);key=(c.doc_id,c.text_sha256,c.char_start,c.char_end)
        passed=not use_reranker or not profile.threshold_enabled or h.rerank_score>=profile.rerank_threshold
        event={'source_id':c.source_id,'original_rank':origins.get(c.source_id,(None,None))[0],
               'fusion_score':origins.get(c.source_id,(None,None))[1],'rerank_rank':rank,'rerank_score':h.rerank_score,
               'threshold_pass':passed,'top_k_pass':None,'dedup_pass':None,'chars':n,'reference_tokens':t,
               'reason':None,'included':False,'source_range':[c.char_start,c.char_end],'page':c.page_from}
        if not passed:event['reason']='below_threshold'
        elif key in seen:event.update(reason='duplicate',dedup_pass=False)
        else:
            seen.add(key);eligible+=1;event.update(dedup_pass=True,top_k_pass=eligible<=profile.rerank_limit)
            if eligible>profile.rerank_limit:event['reason']='outside_top_k'
            elif chars+n>profile.pack_chars:event['reason']='char_budget'
            elif tokens+t>profile.pack_reference_tokens:event['reason']='token_budget'
            else:
                packed.append(h);chars+=n;tokens+=t;event.update(reason='included',included=True)
        events.append(event)
    present={h.chunk.source_id for h in ranked}
    for i,h in enumerate(candidates,1):
        if h.chunk.source_id not in present:events.append({'source_id':h.chunk.source_id,'original_rank':i,'reason':'missing_reranker_result','included':False})
    return packed,{'selection_events':events,'pack_chars':chars,'pack_reference_tokens':tokens,
                   'truncation':'none; whole chunks only','threshold_enabled':profile.threshold_enabled}
