"""Budgeted cold-start and inference windows for a subsequent reader deployment.

Planning only: this cannot extend a worker whose deadline is already fixed.
"""
from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class ReaderTimePlan:
    cold_start_s: int
    inference_s: int
    cleanup_s: int
    required_s: int
    affordable_s: int
    session_s: int
    reservation_usd: float
    feasible: bool
    reason: str

    def receipt(self):
        return asdict(self)


def plan_reader_time(*, remaining_usd, hourly_ceiling, remaining_calls,
                     per_call_s=180, observed_cold_start_s=26*60,
                     minimum_ready_s=30*60, cleanup_s=120,
                     remaining_authorization_s=6*3600):
    """Refuse a knowingly truncated wave instead of spending its ready window loading.

    remaining_usd excludes all existing reservations, including storage/provider
    allowances. The caller must atomically reserve before creating any resource.
    Per-call time is a planning assumption, not a guarantee or proxy-timeout fix.
    """
    values = [remaining_usd, hourly_ceiling, remaining_calls, per_call_s,
              observed_cold_start_s, minimum_ready_s, cleanup_s,
              remaining_authorization_s]
    if any(not math.isfinite(x) or x < 0 for x in values):
        raise ValueError('Finite nonnegative time/budget inputs required')
    if hourly_ceiling <= 0 or per_call_s <= 0 or int(remaining_calls) != remaining_calls:
        raise ValueError('Positive hourly/request bounds and integer call count required')
    cold = math.ceil(max(30*60, observed_cold_start_s*1.15))
    inference = math.ceil(max(minimum_ready_s, remaining_calls*per_call_s))
    cleanup = math.ceil(cleanup_s)
    required = cold + inference + cleanup
    affordable = max(0, math.floor(min(remaining_usd*3600/hourly_ceiling,
                                      remaining_authorization_s)))
    feasible = required <= affordable
    return ReaderTimePlan(cold, inference, cleanup, required, affordable,
                          required if feasible else 0,
                          required*hourly_ceiling/3600 if feasible else 0,
                          feasible, 'ready_window_reserved' if feasible else
                          'insufficient_budget_or_authorization_for_complete_window')
