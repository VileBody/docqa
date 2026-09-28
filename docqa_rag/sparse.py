"""Actual Okapi BM25. Corpus-scoped IDF, frozen vocabulary, no neural sparse model."""
import math,re
from collections import Counter
from qdrant_client.models import SparseVector
TOKENIZER = 'unicode-word-or-identifier-casefold-v1'

def tokens(text):
    return re.findall(r'[\w]+(?:[-/.,][\w]+)*',text.casefold(),flags=re.UNICODE)

class BM25:
    def __init__(self, texts=None, state=None, previous_vocabulary=None):
        if state is not None: self.state=state; return
        counts=[Counter(tokens(t)) for t in texts]; df=Counter(t for c in counts for t in c)
        n=len(counts)
        vocabulary=dict(previous_vocabulary or {})
        if set(vocabulary.values())!=set(range(len(vocabulary))):raise ValueError('Invalid vocabulary dimensions')
        for term in sorted(set(df)-set(vocabulary)):vocabulary[term]=len(vocabulary)
        self.state={'vocabulary':vocabulary,
            'idf':{t:math.log(1+(n-f+0.5)/(f+0.5)) for t,f in df.items()},
            'avgdl':sum(sum(c.values()) for c in counts)/max(n,1),'k1':1.2,'b':0.75,'tokenizer':TOKENIZER,'count':n}
    def vector(self,text,query=False):
        s=self.state; counts=Counter(tokens(text)); dl=sum(counts.values()); pairs=[]
        for token,tf in counts.items():
            if token not in s['vocabulary'] or token not in s['idf']: continue
            value=s['idf'][token] if query else tf*(s['k1']+1)/(tf+s['k1']*(1-s['b']+s['b']*dl/max(s['avgdl'],1)))
            pairs.append((s['vocabulary'][token],value))
        pairs.sort()
        return SparseVector(indices=[p[0] for p in pairs],values=[p[1] for p in pairs])
