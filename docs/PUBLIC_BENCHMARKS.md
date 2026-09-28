# Public benchmark appendix

Historical frozen English pilots with earlier readers/prompts; these scores do not measure the selected Sol/P1 v1/v2 or the final HTTP product. No rerun in this finalization.

| Pilot | Score | Execution |
|---|---|---|
| ContractNLI / Qwen | label accuracy 199/340 = 58.53%; macro-F1 56.11% | 336 native completions |
| ContractNLI / Sosana Luna alias | label accuracy 247/340 = 72.65%; macro-F1 67.92% | 338 native completions |
| IIRC / flow Qwen | EM .29; F1 .317 | 98/100 complete |
| IIRC / static Qwen | EM .25; F1 .2703 | 87/100 complete |
| IIRC / agent | EM null; F1 null; status blocked | planned 100; executed 0 |

Classification accuracy is not general QA accuracy. Exact citation validity does not establish semantic support. Sosana upstream identity is unverified. Fixed-denominator execution failures retain zero; wholly blocked comparisons are N/A. No benchmark rerun during finalization.

The source pilots and evaluator data are excluded from the public release. In the full research checkout, the versioned active handoff is `reports/public_benchmarks/ACTIVE_HANDOFF.json`, canonical results are `reports/final_acceptance/v1/CANONICAL_PUBLIC_RESULTS.json`, and raw report/judgments remain in `reports/public_benchmarks/v1`.

The historical Stage 3 research is only partially complete: Qwen controller gate blocked, authored reserved is not independent blind testing, and all eight dependency fixtures also admitted a sufficient first flow pack. This does not establish a benefit from dependent second search.
