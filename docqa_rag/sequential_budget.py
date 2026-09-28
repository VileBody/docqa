"""Per-episode attempt guard around the existing monetary Calls ledger.

The parent may retry transport failures. Guard runs inside its callback, before
EVERY actual HTTP attempt, so retry work also consumes the episode ceilings.
"""
import json
import time
from dataclasses import dataclass
from .util import BudgetExceeded


@dataclass(frozen=True)
class AttemptLimits:
    requests: int = 64
    rerank_pairs: int = 72
    output_tokens: int = 8192

    def __post_init__(self):
        if any(type(v) is not int or v <= 0 for v in vars(self).values()):
            raise ValueError('Positive integer attempt ceilings required')


class EpisodeCalls:
    def __init__(self, parent, limits, deadline):
        self.parent = parent
        self.limits = limits
        self.deadline = deadline
        self.actual = {'requests': 0, 'rerank_pairs_submitted': 0, 'output_token_reserve': 0}
        self.blocked = None

    @property
    def records(self):
        return self.parent.records

    def invoke(self, kind, text, fn, output_tokens=0, request_timeout_s=30):
        pairs = len(json.loads(text)['documents']) if kind == 'reranker' else 0
        def guarded():
            proposed = {'requests': self.actual['requests'] + 1,
                        'rerank_pairs_submitted': self.actual['rerank_pairs_submitted'] + pairs,
                        'output_token_reserve': self.actual['output_token_reserve'] + output_tokens}
            if self.blocked:
                raise BudgetExceeded(self.blocked)
            if time.monotonic() + request_timeout_s > self.deadline:
                self.blocked = 'timeout'
                raise BudgetExceeded(self.blocked)
            if (proposed['requests'] > self.limits.requests or proposed['rerank_pairs_submitted'] > self.limits.rerank_pairs
                    or proposed['output_token_reserve'] > self.limits.output_tokens):
                self.blocked = 'budget_exhausted'
                raise BudgetExceeded(self.blocked)
            self.actual = proposed
            return fn()
        return self.parent.invoke(kind, text, guarded, output_tokens, request_timeout_s)
