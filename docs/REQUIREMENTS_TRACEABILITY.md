# Соответствие ТЗ — Sol/P1 v2

Текущая доработка — retry configuration и публичная поставка. Новых live-вызовов нет. Исторические Sol v1 HTTP12/12 (одна попытка) не переименованы в новый прогон.

| ID | Статус | Доказательство и границы |
|---|---|---|
| T01 | PASS | FastAPI routes and request/response models [Historical infrastructure evidence plus unchanged runtime; not a new live execution.] |
| T02 | PASS | Queued upload and durable task identity [Historical infrastructure evidence plus unchanged runtime; not a new live execution.] |
| T03 | PASS | Unknown task 404, ready/failed states [Historical infrastructure evidence plus unchanged runtime; not a new live execution.] |
| T04 | PASS | Catalog pending/failed/ready; idempotency [Historical infrastructure evidence plus unchanged runtime; not a new live execution.] |
| T05 | PASS | doc_id/snapshot/ready filtering and exact citation scope [Historical infrastructure evidence plus unchanged runtime; not a new live execution.] |
| T06 | PASS | New cold upload of existing 100-logical-page TXT; real Celery indexing; exact page-100 answer. Small functional fixture, not throughput. [Historical infrastructure evidence plus unchanged runtime; not a new live execution.] |
| T07 | PASS | Historical Sol/P1 v1 HTTP: 12/12 semantic checks, H03 correctly not_found with nonempty pack; normal paraphrase and explicit equivalence preserved. V2 changes only retries; no new live quality score. |
| T08 | PASS | Historical Sol/P1 v1: H03/H09/H11/H12 correct not_found, explicit prohibition and calculation retained. Strict states and authentic spans covered by public CPU tests; v2 transport does not reroll semantic results. |
| T09 | PASS | H11/H12 actual P1 model inputs equal, 14 messages, 6 fixed teaching pairs, no previous user history; two provider calls. [Historical infrastructure evidence plus unchanged runtime; not a new live execution.] |
| T10 | PASS | Real Celery/Redis including worker SIGKILL [Historical infrastructure evidence plus unchanged runtime; not a new live execution.] |
| T11 | PASS | Shared generation collection and doc filter [Historical infrastructure evidence plus unchanged runtime; not a new live execution.] |
| T12 | PASS | Qdrant document payload; Redis task state [Historical infrastructure evidence plus unchanged runtime; not a new live execution.] |
| T13 | PASS | Chunk payload and offsets validated on publication/restore [Historical infrastructure evidence plus unchanged runtime; not a new live execution.] |
| T14 | PASS | doc_id/snapshot/ready payload indexes [Historical infrastructure evidence plus unchanged runtime; not a new live execution.] |
| T15 | PASS | dense and bm25 in each point [Historical infrastructure evidence plus unchanged runtime; not a new live execution.] |
| T16 | PASS | Frozen BM25 statistics and append vocabulary rebuild [Historical infrastructure evidence plus unchanged runtime; not a new live execution.] |
| T17 | PASS | One primary native fusion request; diagnostic queries separate [Historical infrastructure evidence plus unchanged runtime; not a new live execution.] |
| T18 | PASS | Validated env fusion/input limits [Historical infrastructure evidence plus unchanged runtime; not a new live execution.] |
| T19 | PASS | 12 actual separately priced-in-GPU reranker calls; confirmed explicit-instruction-v2 worker, candidates and pack trace. [Historical infrastructure evidence plus unchanged runtime; not a new live execution.] |
| T20 | PASS | Pack only from ranked authentic candidates; bounded top-k [Historical infrastructure evidence plus unchanged runtime; not a new live execution.] |
| T21 | PASS | Env reranker model/revision/input/output limits wired [Historical infrastructure evidence plus unchanged runtime; not a new live execution.] |
| T22 | PASS | Historical Sol/P1 v1: 12 actual LangChain reader completions, 5 restored documents + 2 existing fixtures; full logical requests verified. V2 first-attempt messages/schema/parameters unchanged, tested locally. |
| T23 | PASS | Pydantic contracts and invalid citation/model rejection [Historical infrastructure evidence plus unchanged runtime; not a new live execution.] |
| T24 | PASS | Profile defaults/env overrides, complete example config [Historical infrastructure evidence plus unchanged runtime; not a new live execution.] |
| T25 | PASS | V2 actual Calls factory: at most 3 attempts total on classified transient transport/408/429/500/502/503/504 errors; 503 then success, permanent/invalid errors once, three failures stop, budget/deadline blocks another send. Each attempt retained in ledger/records. CPU MockTransport, not live fault-test. |
| T26 | PASS | Observed same-snapshot branch replay and separate query cost [Historical infrastructure evidence plus unchanged runtime; not a new live execution.] |
| T27 | PASS | V2 explicit allowlist ZIP with hashes/CRC and available-secret scan; 42/42 self-contained public tests pass on host, isolated release and clean Linux image with network disabled. Historical cleanrooms remain separate: Luna 30 tests; Sol v1 14 tests. No new paid acceptance. |
| T28 | PASS | Architecture, limits, production gaps documented [Historical infrastructure evidence plus unchanged runtime; not a new live execution.] |
| T29 | PASS | Actual exact identifier H01 and nonempty no-answer H11 published unedited in API_EXAMPLES; H03 failure also retained. [Historical infrastructure evidence plus unchanged runtime; not a new live execution.] |

[Текущие синтетические JSON-примеры](submission/evidence/SYNTHETIC_HTTP_OUTPUTS.json) · [точные версии и provenance](REQUIREMENTS_TRACEABILITY.json). Исторические Luna H03 failure и cleanroom30, Sol v1 cleanroom14 сохранены раздельно.
