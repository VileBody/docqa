"""Stage 3 library retrieval entry point; application owns index/scope/budget.

No resource creation, gold access, synthesis, automatic fallback or paid retry
wave. Calls may contact configured models: caller must supply an authorized ledger.
"""
import time
from .adapters import QwenEmbeddings, QwenReranker, ChatAdapter
from .episode_calls import EpisodeCalls, AttemptLimits
from .search_evidence import EvidenceSearch, SearchLimits, InvalidToolRequest
from .search_modes import retrieve_mode
from .sequential_modes import retrieve_sequential, ControllerLimits
from .util import BudgetExceeded, ModelOutputInvalid, ModelUnavailable

VERSION = 'stage3-library-v2'
MODES = ('flow', 'tool_once', 'static_batch', 'reactive', 'research', 'flow_wide')
DEFAULT_LIMITS = SearchLimits(searches=2, views=2, rerank_pairs=72, wall_seconds=180)
DEFAULT_CONTROLLER = ControllerLimits(max_calls=3, total_output_token_reserve=3072)


def _trace(trace, guard):
    value = dict(trace)
    if 'wire_attempts' in value:
        value['legacy_attempt_counters'] = value.pop('wire_attempts')
    value['wire_accounting'] = 'callback_reservations_not_confirmed_http'
    value.update(library_version=VERSION, attempt_accounting=guard.accounting())
    return value


def retrieve_evidence(index, profile, calls, *, doc_id, question, mode='flow',
                      limits=DEFAULT_LIMITS, controller_limits=DEFAULT_CONTROLLER,
                      attempt_limits=AttemptLimits(), checkpoint=None):
    """Return authentic final hits and a public trajectory. Synthesis is separate.

    Requires a restored index, explicit Profile and Calls with an existing budget.
    The default remains flow. Experimental modes use the pinned Qwen controller.
    Exceptions become operational statuses, never ``not_found`` answers.
    """
    if mode not in MODES:
        raise ValueError('Unknown stage3 mode')
    if doc_id not in index.documents or not question.strip():
        raise InvalidToolRequest('Application document scope/question missing')
    if attempt_limits.rerank_pairs > limits.rerank_pairs:
        raise ValueError('Attempt pair ceiling exceeds tool ceiling')
    if limits.searches not in (1, 2, 4, 8):
        raise ValueError('Predeclare 1, 2, 4 or 8 searches')
    effective = profile.model_copy(deep=True)
    if mode == 'flow_wide':
        effective = effective.model_copy(update={'candidate_limit': 48})
    guard = EpisodeCalls(calls, attempt_limits, time.monotonic() + limits.wall_seconds)
    tool = None
    trace = {'mode': mode, 'original_question': question}
    pack = []
    def save(value):
        if checkpoint:
            checkpoint(_trace(value, guard))
    try:
        embedder = QwenEmbeddings(effective.embedding, guard)
        reranker = QwenReranker(effective.reranker, guard)
        tool = EvidenceSearch(index, embedder, reranker, effective,
                              doc_id=doc_id, question=question, limits=limits)
        controller = None
        if mode not in ('flow', 'flow_wide'):
            controller = ChatAdapter(effective.generators['qwen'], guard,
                max_tokens=controller_limits.output_tokens_per_call, enforce_context_limits=True)
        if mode in ('reactive', 'research'):
            pack, trace = retrieve_sequential(tool, mode, controller,
                                              limits=controller_limits, checkpoint=save)
        else:
            pack, trace = retrieve_mode(tool, 'flow' if mode == 'flow_wide' else mode, controller)
            trace['mode'] = mode
        if controller:
            trace['controller_usage'] = controller.usage
    except (BudgetExceeded, ModelOutputInvalid, ModelUnavailable, InvalidToolRequest, TimeoutError) as exc:
        status = next(label for cls, label in (
            (BudgetExceeded, 'budget_exhausted'), (ModelOutputInvalid, 'model_output_invalid'),
            (ModelUnavailable, 'model_unavailable'), (InvalidToolRequest, 'invalid_tool_request'),
            (TimeoutError, 'timeout')) if isinstance(exc, cls))
        trace.update(status=status, stop_reason=status, error_type=type(exc).__name__)
    finally:
        if tool:
            trace.update(searches=tool.history, actual=dict(tool.actual))
        if guard.blocked:
            trace.update(status=guard.blocked, stop_reason=guard.blocked)
            pack = []
        if trace.get('status') != 'success':
            pack = []
        save(trace)
    return pack, _trace(trace, guard)
