# Sol/P1 submission v2 — перед локальной передачей

Доработка ограничена retry configuration/tests, согласованием документации и публичной поставкой. Runtime reader, P1 с шестью демонстрациями, SpanDraft, strict renderer, binding/reasoning и retrieval не изменены. Старые outputs/freezes/ledger сохранены.

## Текущая версия и проверки

`profiles/submission/sol_p1_v2.json`: version=submission-sol-p1-v2, max_attempts=3. Это максимум три попытки всего при уже классифицированных transient transport/408/429/500/502/503/504 ошибках. SDK retries=0; общий Calls сохраняет бюджет/deadline/backoff. JSON/evidence/semantic/permanent auth/config ошибки не вызывают reroll. Неоднозначный timeout может означать повторную оплату; все попытки сохраняются и резервируются отдельно.

**25/25 новых CPU retry-проверок**: фактическая фабрика Calls, HTTP503→успех для reader/embedding/reranker; временные статусы и timeout; постоянные ошибки; невалидный ответ/источник; semantic FAIL; три неудачи; лимиты денег/числа вызовов/deadline; совпадение first-attempt request identity с v1. Это MockTransport, не live fault-test. **42/42 public tests** прошли на host, из отдельной release-папки и в чистом Linux-образе с `--network none`; наборы пересекаются и не складываются. Статический tokenizer cache смонтирован read-only из ранее проверенного локального файла. Новый Docker build успешен. Проверка чистой упаковки отражена в `submission/evidence/SUBMISSION_V2_VALIDATION.json`.

## Исторические результаты — не новые прогоны v2

| Проверка | Версия | Результат |
|---|---|---|
| Известные reader случаи + повторы | Astra и Sol/P1, stronger_reader_v1 | по 40/40; по 38 новых eval-generations,2 пустых pack локально |
| Штатный API с retrieval | Sol/P1 v1, max_attempts=1 |12/12; H03 not_found; full request/цитаты проверены |
| Общая CPU-регрессия | Sol v1 checkout |712 passed / 8 skipped;5 historical freeze tests отдельно исключены |
| Public cleanroom CPU | Sol v1 |14 passed; холодный tokenizer asset требует сети/кэша |
| Infrastructure cleanroom | Luna historical |30 passed; не текущий v2 результат |
| Прежний HTTP | Luna/P1 |11/12; H03 failure сохранён |

Смысловое ревью агентное; независимого человеческого ревью нет. Наборы известные/authored, не blind-test. Исследовательские [ContractNLI/IIRC](PUBLIC_BENCHMARKS.md) принадлежат прежним profiles и не являются оценкой Sol.

## Публичные свидетельства

[9 синтетических HTTP-примеров](submission/evidence/SYNTHETIC_HTTP_OUTPUTS.json) извлечены из сохранённого Sol v1 результата. Тексты ответов и цитат совпадают с raw; служебные/private поля удалены с описанием преобразования. Исторический Luna-файл сохранён отдельно в research checkout; provenance доступен в `submission/evidence/HISTORICAL_LUNA_EXAMPLES_METADATA.json`. Публичные примеры не содержат real TXT, gold, полных traces или reviewer rubrics.

T07/T08/T22/T25/T27 в [MD](REQUIREMENTS_TRACEABILITY.md) и [JSON](REQUIREMENTS_TRACEABILITY.json) согласованы по версии и уровню доказательства. Остальная инфраструктура подтверждается явно историческими проверками, без заявления о повторе всех стадий.

## Поставка и запуск

Единственный основной quickstart — [README](../README.md), подключения — [MODEL_ENDPOINTS](MODEL_ENDPOINTS.md). Удалённые Pods уничтожены; прежние URL не считаются работающими. Compose ничего не арендует. `.env.example` включает отключённые модельные вызовы; бюджет для будущего запуска задаёт оператор.

Новый ZIP создаёт `scripts/build_sol_release.py` из явного allowlist: код, profiles/P1, зависимости, Docker/Compose, поддерживаемые tests, свои небольшие demo, tokenizer assets/licence, актуальные документы и ограниченные свидетельства. Исключены secrets, old .git, weights, vectors/snapshots, full runs, evaluator/gold data, reviewer rubrics, private traces, storage dumps и агентские задания. Проверка доступных env/dotenv-значений, private paths, hashes/CRC не является обещанием абсолютной безопасности или аудитом Git history. Owner bundle остаётся локальным.

Статус: **READY_LOCAL_DELIVERY** после локальных checks и сборки. Новых model/GPU расходов0. Нет production SLA или гарантии абсолютной фактической точности. Автоматическая публикация GitHub/отправка не выполнена; после поставки остановка.
