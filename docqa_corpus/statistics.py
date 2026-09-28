"""Paired family-level sensitivity for already scored RAG outputs (Python 3.10+).

This is not a semantic judge, a RAG runner, or a population guarantee. Input
scores must be independently obtained using the frozen rubric. Bootstrap
resamples observed families, not questions, generations, or mutation copies.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable


def percentile(values: list[float], probability: float) -> float:
    if not values or not 0 <= probability <= 1:
        raise ValueError("A nonempty sample and probability in [0,1] are required")
    values = sorted(values)
    position = (len(values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    weight = position - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def zero_failure_upper(n: int, confidence: float = 0.95) -> float:
    """Exact one-sided binomial upper bound given ZERO failures.

    ASSUMPTION: n independent Bernoulli trials with a common failure probability.
    Do not substitute clustered RAG question count for n without justification.
    """
    if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0,1)")
    return -math.expm1(math.log1p(-confidence) / n)


def binary_mix_score(answerable_score: float, unanswerable_score: float,
                     unanswerable_share: float) -> float:
    """Hypothetical mixture, not an estimate of the prevalence in real traffic."""
    values = [answerable_score, unanswerable_score, unanswerable_share]
    if any(not math.isfinite(x) or not 0 <= x <= 1 for x in values):
        raise ValueError("Both conditional scores and the share must be in [0,1]")
    return ((1 - unanswerable_share) * answerable_score
            + unanswerable_share * unanswerable_score)


def paired_family_report(records: Iterable[dict[str, Any]], system_a: str,
                         system_b: str, *, suite: str = "S", split: str = "dev",
                         metric: str = "task_success", resamples: int = 5000,
                         seed: int = 20260924) -> dict[str, Any]:
    """Difference is B minus A. Scores must be finite values in [0,1].

    Aggregation: repetitions -> question -> family -> equal family average.
    M is intentionally excluded: score metamorphic relations separately.
    No missing-case deletion or silently dropped failed calls is performed.
    """
    if system_a == system_b:
        raise ValueError("Select two different systems")
    if suite in ("M", "C"):
        raise ValueError("M/C are separate behavioral suites, not primary accuracy pools")
    if isinstance(resamples, bool) or not isinstance(resamples, int) or resamples < 100:
        raise ValueError("Use an integer >=100 bootstrap resamples")
    selected = [r for r in records if r.get("system_id") in (system_a, system_b)
                and r.get("suite") == suite and r.get("split") == split]
    if not selected:
        raise ValueError("No rows match the requested systems/suite/split")
    # Never combine incompatible snapshots, rubric versions or evaluation pools.
    provenance = {}
    for key in ("dataset_hash", "gold_version", "eval_pool"):
        vals = {r.get(key) for r in selected}
        if len(vals) != 1 or None in vals or "" in vals:
            raise ValueError(f"Exactly one nonempty {key} must be shared by all rows")
        provenance[key] = next(iter(vals))
    by_system: dict[str, dict[tuple[str, str], dict[str, Any]]] = {
        system_a: {}, system_b: {}}
    for r in selected:
        if not all(isinstance(r.get(k), str) and r[k] for k in
                   ("question_id", "family_id", "repeat_id")):
            raise ValueError("question_id, family_id and repeat_id must be nonempty strings")
        score = r.get(metric)
        if isinstance(score, bool):
            score = float(score)
        if not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError(f"Missing/invalid numeric {metric}: {r.get('question_id')}")
        if metric == "task_success" and score not in (0, 1):
            raise ValueError("task_success must be binary 0/1 before averaging repetitions")
        if r.get("status") not in ("ok", "error"):
            raise ValueError("status must be ok or error")
        if metric == "task_success" and r["status"] == "error" and score != 0:
            raise ValueError("Service errors must have task_success=0")
        key = (r["question_id"], r["repeat_id"])
        if key in by_system[r["system_id"]]:
            raise ValueError(f"Duplicate case/repeat for {r['system_id']}: {key}")
        by_system[r["system_id"]][key] = {**r, metric: float(score)}
    a, b = by_system[system_a], by_system[system_b]
    if not a or set(a) != set(b):
        raise ValueError("Unpaired cases/repeats; record errors as rows, never drop them")
    question_values: dict[str, list[tuple[float, float]]] = defaultdict(list)
    question_family: dict[str, str] = {}
    question_repeats: dict[str, set[str]] = defaultdict(set)
    for key in sorted(a):
        ar, br = a[key], b[key]
        if ar["family_id"] != br["family_id"]:
            raise ValueError(f"Family disagreement: {key}")
        question = key[0]
        family = ar["family_id"]
        if question in question_family and question_family[question] != family:
            raise ValueError("Family must be stable across repetitions")
        question_family[question] = family
        question_repeats[question].add(key[1])
        question_values[question].append((ar[metric], br[metric]))
    if len({frozenset(v) for v in question_repeats.values()}) != 1:
        raise ValueError("Every question must have the same predeclared repetition IDs")
    families: dict[str, list[tuple[float, float]]] = defaultdict(list)
    questions = []
    for q, vals in question_values.items():
        av, bv = fmean(v[0] for v in vals), fmean(v[1] for v in vals)
        families[question_family[q]].append((av, bv))
        questions.append((av, bv))
    if len(families) < 2:
        raise ValueError("At least two families are required; two are still not reliable inference")
    family_stats = []
    for family, vals in sorted(families.items()):
        av, bv = fmean(v[0] for v in vals), fmean(v[1] for v in vals)
        family_stats.append({"family_id": family, "question_count": len(vals),
                             "score_a": av, "score_b": bv, "delta_b_minus_a": bv - av})
    deltas = [f["delta_b_minus_a"] for f in family_stats]
    delta = fmean(deltas)
    rng = random.Random(seed)
    bootstrap = [fmean(rng.choices(deltas, k=len(deltas))) for _ in range(resamples)]
    lodo = []
    for idx, f in enumerate(family_stats):
        remaining = fmean(d for i, d in enumerate(deltas) if i != idx)
        lodo.append({"omitted_family": f["family_id"], "delta_b_minus_a": remaining,
                     "change_from_full": remaining - delta,
                     "strict_sign_flip": delta * remaining < 0})
    warnings = [
        "Exploratory percentile bootstrap conditional on observed families; not post-selection-adjusted.",
        "Family IDs are a grouping convention, not proof of independent/representative sampling.",
        "Leave-one-family-out is influence analysis on frozen outputs, not cross-validation.",
        "A confidence interval containing zero does NOT establish equivalence.",
        "The positive bootstrap fraction is NOT a posterior probability that B is better.",
    ]
    if len(families) < 20:
        warnings.append("Small number of families (<20 is a heuristic flag): interval and ranks may be unstable.")
    if min(deltas) == max(deltas):
        warnings.append("All family deltas identical: bootstrap is degenerate, not proof of no uncertainty.")
    return {"metric": metric, "system_a": system_a, "system_b": system_b,
            "suite": suite, "split": split, **provenance,
            "questions": len(questions), "families": len(family_stats),
            "repetitions_per_question": len(next(iter(question_repeats.values()))),
            "macro_a": fmean(f["score_a"] for f in family_stats),
            "macro_b": fmean(f["score_b"] for f in family_stats),
            "micro_a": fmean(v[0] for v in questions),
            "micro_b": fmean(v[1] for v in questions),
            "delta_b_minus_a": delta,
            "exploratory_percentile_interval_95": [percentile(bootstrap, 0.025), percentile(bootstrap, 0.975)],
            "bootstrap_fraction_delta_positive": fmean(x > 0 for x in bootstrap),
            "bootstrap_resamples": resamples, "bootstrap_seed": seed,
            "family_scores": family_stats, "leave_one_family_out": lodo,
            "warnings": warnings}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="JSONL scored outputs, one row per case/repeat/system")
    parser.add_argument("--a", required=True)
    parser.add_argument("--b", required=True)
    parser.add_argument("--suite", default="S", choices=["S", "R", "N"])
    parser.add_argument("--split", default="dev")
    parser.add_argument("--metric", default="task_success")
    parser.add_argument("--resamples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    result = paired_family_report(rows, args.a, args.b, suite=args.suite, split=args.split,
                                  metric=args.metric, resamples=args.resamples, seed=args.seed)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
