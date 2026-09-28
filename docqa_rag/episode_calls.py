"""Maintained attempt accounting; frozen sequential_budget remains replayable.

An admitted callback is not proof of HTTP dispatch or server computation.
The inherited ``actual`` mapping is retained only for legacy mode compatibility.
"""
from .sequential_budget import EpisodeCalls as LegacyEpisodeCalls, AttemptLimits

VERSION = 'episode-callback-accounting-v2'


class EpisodeCalls(LegacyEpisodeCalls):
    @property
    def budget(self):
        """Forward the same monetary budget used by all real adapters."""
        return self.parent.budget

    def __init__(self, parent, limits: AttemptLimits, deadline):
        super().__init__(parent, limits, deadline)
        self.events = []

    def invoke(self, kind, text, fn, output_tokens=0, request_timeout_s=30):
        def observed():
            event = {'kind': kind, 'outcome': 'started'}
            self.events.append(event)
            try:
                result = fn()
            except BaseException as exc:
                # Exception messages, request text and credentials are not telemetry.
                event.update(outcome='raised', error_type=type(exc).__name__)
                raise
            event['outcome'] = 'returned'
            return result
        return super().invoke(kind, text, observed, output_tokens, request_timeout_s)

    def accounting(self):
        return {
            'version': VERSION,
            'callback_attempts': self.actual['requests'],
            'rerank_pairs_reserved': self.actual['rerank_pairs_submitted'],
            'output_tokens_reserved': self.actual['output_token_reserve'],
            'callbacks_returned': sum(e['outcome'] == 'returned' for e in self.events),
            'callbacks_raised': sum(e['outcome'] == 'raised' for e in self.events),
            'guard_stop': self.blocked,
            'http_dispatches': None,
            'server_completed_pairs': None,
            'transport_observability': 'not_instrumented_do_not_infer_from_callbacks',
            'events': [dict(e) for e in self.events],
        }
