# DocQA: ответы по одному TXT

Сервис принимает UTF-8 TXT, индексирует его в фоне и отвечает на независимые вопросы по выбранному документу. Текущий профиль — **submission-sol-p1-v2**; reader — **Sosana/Sol (`gpt-6-sol`), P1, шесть фиксированных демонстраций, `strict_answerability_v1`**. Это alias внешнего провайдера; upstream OpenAI не подтверждён.

Состояние поставки: **READY_LOCAL_DELIVERY**. Исторический Sol/P1 v1 прошёл **12/12 HTTP-проверок**, включая H03 (max_attempts=1). В v2 изменены только retry configuration и публичная упаковка: максимум 3 попытки всего при временных сбоях; новых платных вызовов нет. На известных reader-regression случаях Astra и Sol прошли 40/40; выбран Sol по заранее заданному правилу задержки/стоимости. Исторический результат Luna 11/12 и неуспешный checker сохранены отдельно. Это ограниченная приёмка с агентным ревью, не гарантия общей точности. См. [отчёт](docs/FINAL_REPORT.md), [требования](docs/REQUIREMENTS_TRACEABILITY.md), [ответы API](docs/API_EXAMPLES.md).

## Архитектура и границы

FastAPI → Redis-каталог/outbox → Celery → LangChain splitter/embeddings → Qdrant. В каждой опубликованной генерации одна общая коллекция для всех готовых документов. Chunk-point содержит dense Qwen и настоящий sparse BM25, текст, документ, страницы и смещения. Один основной Qdrant Query API объединяет dense/BM25 через native fusion с фильтрами `doc_id/snapshot/ready`. Отдельный Qwen reranker оценивает кандидатов; в Sol попадает только выбранный pack. Публичные цитаты разрешаются по текущему TXT, а не по учебным источникам.

Redis атомарно публикует готовую неизменяемую генерацию. Pending/failed остаются в каталоге; незавершённый документ недоступен для QA. Повторная доставка задачи идемпотентна. Новые слова входят в перестроенный BM25; совместимые dense-векторы переиспользуются. Reader/config не меняет embedding identity нового submission-профиля. Старые snapshots с прежним pipeline требуют отдельного проверенного переноса/восстановления и не подключаются молча.

TXT — источник истины. До 100 логических страниц через `\f`, иначе одна страница; до 30 MiB. Смещения — символы Unicode в неизменном декодированном тексте. BOM/NUL/невалидный UTF-8 отклоняются; пустые страницы сохраняются. OCR и восстановление PDF не входят в поставку. TXT-only — принятое решение реализации; отдельное согласие работодателя на него не подтверждено.

## Запуск

Нужны Docker Engine/Desktop с Compose v2, Python 3.12 для клиентских команд и доступ к модельным HTTPS endpoints. Все команды ниже выполняются из корня этого комплекта. Compose не арендует GPU. Порт API по умолчанию 18080.

```sh
cp .env.example .env
# Заполните DOCQA_API_KEY (не менее 16 символов) и модельные подключения.
docker compose --env-file .env -f profiles/stage_2/compose.yaml up --build -d
curl http://127.0.0.1:18080/health
```

`/health` подтверждает процесс и режим, а не готовность моделей. Для реальных ответов нужны:

- `DOCQA_EMBED_BASE_URL/API_KEY`: Qwen3-Embedding-0.6B, 1024 измерения, revision из `profiles/stage_2/txt.json`.
- `DOCQA_RERANK_BASE_URL/API_KEY`: Qwen3-Reranker-0.6B, закреплённый revision, контракт `explicit-instruction-v2` с подтверждением SHA инструкции и оценкой каждого кандидата.
- `DOCQA_OPENAI_BASE_URL/API_KEY`: совместимый Chat Completions endpoint Sosana/Sol. Эти имена — технический слот адаптера, не утверждение об OpenAI.

Проверенная последовательность запуска и обновления временных endpoints — в [MODEL_ENDPOINTS](docs/MODEL_ENDPOINTS.md). Модельные endpoints обслуживает оператор. Можно использовать существующий совместимый сервер; для временного RunPod сохранён проверенный worker `scripts/qwen_worker_v2.py` и lifecycle `docqa_rag.lifecycle.Session`. Worker требует owner session/token, hard deadline и heartbeat, не запускается отдельной командой без этих параметров. Аренда выполняется только после нового разрешения; URL удалённого Pod не является рабочим endpoint. Проверяйте `/health` worker: точные model revisions, hardware и rerank contract; затем укажите актуальный `/v1` URL в `.env` и пересоздайте API/worker. Проверенная упаковка дополнительно загружает Qwen4B, но reader в submission — Sol; этот overhead не скрывается.

По умолчанию `DOCQA_MODEL_CALLS_ENABLED=0`. После разрешения бюджета создайте ledger в общем `/data` и включите calls:

```sh
# Числа являются примером команды, НЕ разрешением на расходы.
docker compose --env-file .env -f profiles/stage_2/compose.yaml exec -T api \
  python -c 'from docqa_rag.budget import Budget; Budget("/data/authorized-budget.json",2.5,256,10800)'
# В .env: DOCQA_MODEL_CALLS_ENABLED=1, затем:
docker compose --env-file .env -f profiles/stage_2/compose.yaml up -d
```

Ledger ограничивает денежные резервы/число попыток/срок. Аренду GPU оператор учитывает в том же общем лимите, отдельно от вызовов. Исторические остатки не означают разрешение. Тарифы в `profiles/submission/sol_prices.json` — зафиксированные ранее значения, их нужно подтвердить перед новой серией.

## HTTP

Установите локальный `DOCQA_API_KEY` в shell, не публикуя его. Загрузите собственный demo:

```sh
curl -H "Authorization: Bearer $DOCQA_API_KEY" -F classification=synthetic \
  -F file=@examples/acceptance/alpha.txt http://127.0.0.1:18080/documents
curl -H "Authorization: Bearer $DOCQA_API_KEY" http://127.0.0.1:18080/tasks/TASK_ID
curl -H "Authorization: Bearer $DOCQA_API_KEY" http://127.0.0.1:18080/documents
curl -H "Authorization: Bearer $DOCQA_API_KEY" -H 'Content-Type: application/json' \
  -d '{"document_id":"DOCUMENT_ID","question":"Какой код используется для отслеживания?","diagnostics":true}' \
  http://127.0.0.1:18080/questions
```

Подставьте возвращённые ID, дождитесь `ready`. Второй независимый вопрос: «Какова стоимость страхования коробки при доставке?». В сохранённом Sol v1 живом прогоне ответ на точный вопрос — «Для отслеживания заявки используется код ALPHA-77.»; на вопрос о страховке — found=false и «Недостаточно сведений в источниках.». Текущие фактические ответы приведены в [API_EXAMPLES](docs/API_EXAMPLES.md) и [публичном JSON](docs/submission/evidence/SYNTHETIC_HTTP_OUTPUTS.json); прежний Luna H03 failure сохранён в историческом отчёте.

Готовый клиент загрузки/poll/list/QA: `scripts/submission_http.py`. Его публичный CPU-plan содержит четыре вопроса; приватный финальный plan — 12, включая два существующих real-dev TXT. Live-клиент требует отдельный authorization с scope и hash входов. В API не передаются эталоны.

## Ответы и ошибки

`complete`: все запрошенные части подтверждены, есть claims, `missing` пуст. `partial`: ответ на самостоятельную часть составного вопроса плюс нейтральное сообщение о неподтверждённом остатке. Произвольный текст `missing` наружу не выводится. Соседнее правило не является partial-ответом на единственную отсутствующую величину. `not_found`, `ambiguous`, `contradictory` возвращают `found=false`, пустые citations и нейтральное сообщение. Явный запрет/ноль — подтверждённый ответ. P1 допускает одну простую арифметическую операцию по правилу источника и явно указанным операндам вопроса.

Точность цитаты доказывает происхождение, но не смысловую поддержку вывода. Critic явно `off`; проверка источников и диапазонов обязательна. Шесть учебных пар не являются пользовательской историей. Кэш генерации отсутствует: каждый непустой pack обрабатывается заново; P0/P1 не разделяют cached ответы.

404 — неизвестный документ/task; 409 — ещё не опубликован; 422 — неверный input/несовместимый профиль; 429 — лимит; 502 — невалидный модельный ответ; 503 — модель, источник или хранилище недоступны. Технический сбой не превращается в `found=false`. Для выбранного Sol v2 — максимум 3 модельные попытки всего (не 3 дополнительных повтора), таймаут каждой 180 с и существующий общий deadline/budget/backoff. Повторяются только уже классифицированные transport/408/429/500/502/503/504 сбои; read retries хранилища сохранены; постоянные ошибки и невалидная схема не повторяются. Неопределённые writes не повторяются вслепую. `POST /tasks/{task_id}/retry` явно повторяет неудавшуюся индексацию.

## Конфигурация и проверки

JSON profile → явно заданные env overrides. Compose передаёт одинаковые настройки API/worker. `DOCQA_READER_PROFILE` включает submission; обычные исторические library profiles сохраняют legacy. Менять модель нужно вместе с revision; выбранный binding защищён hash-проверкой: несовместимые env overrides отклоняются. `DOCQA_RERANK_INPUT_LIMIT=24`, `OUTPUT_LIMIT=6`, `PACK_CHARS=8000`, fusion `rrf`; числа и инструкции проходят до wire. Sol запрашивается с medium reasoning,8192 completion tokens без temperature. Полный P1 prompt, schema и output reserve проверяются без скрытого усечения; для внешнего tokenizer используется консервативная оценка, не billed tokens.

Диагностика повторно измеряет dense/BM25 на том же snapshot и тех же векторах. Это observed replay counts, не счётчики первого fusion. Основные кандидаты берутся только из одного fusion-запроса. При `diagnostics=false` лишние запросы отключены.

```sh
python3 -m venv .venv
.venv/bin/pip install -c requirements-stage2-tested.txt '.[rag,service,dev]'
.venv/bin/python -m pytest tests
```

Отдельный CPU-demo: в `.env` поставьте `DOCQA_TEST_MODE=1`, оставьте `DOCQA_MODEL_CALLS_ENABLED=0` и пересоздайте Compose. Это настоящие API/Celery/Redis/Qdrant, но детерминированные model doubles; качество ответов этим не измеряется.

```sh
.venv/bin/python scripts/submission_http.py --output local_cpu_receipt
docker compose --env-file .env -f profiles/stage_2/compose.yaml down
```

Не удаляйте volumes до export и проверки restore. `python -m docqa_service export /data/snapshot` сохраняет готовый индекс; restore требует пустого namespace и точного fingerprint. Pending/очередь не входят в snapshot. Временную GPU удаляет оператор с проверкой API inventory.

Перед production нужны tenant isolation, TLS/network policy, мониторинг, retention/GC, restore drills и более широкий смысловой контроль. Debug содержит исходные тексты, хранится owner-only с expires_at; используйте `docqa_rag.debug.prune_expired`, автоматического планировщика очистки пока нет. Сервис не заявлен production-ready. Действующий LICENSE сохранён без переопределения; чужие модели/данные имеют собственные условия. В публичный ZIP не входят ключи, веса, gold или приватные traces. Автоматической публикации нет.


История: Luna/P1 HTTP11/12 (H03 failure); answer-fit25/30, не активирован. Эти результаты не исправлялись. Новый Sol profile: `profiles/submission/sol_p1_v2.json`; runtime: `profiles/submission/sol_p1_runtime.json`.

Первый запуск загружает `cl100k_base.tiktoken` из `openaipublic.blob.core.windows.net`; для offline-среды заранее заполните `TIKTOKEN_CACHE_DIR`. Исторический Sol v1 cleanroom:14/14 CPU/integration с model doubles. Текущий v2 public subset: 42/42 на host, из отдельной release-папки и в чистом Linux-образе без сети; результаты фиксируются в `docs/submission/evidence/SUBMISSION_V2_VALIDATION.json`. Платные ключи в тесты не передаются.

При неоднозначном timeout провайдер мог уже списать деньги. Каждая новая попытка отдельно резервирует бюджет и сохраняется в Calls.records/ledger; это не независимый успешный тест. Невалидный JSON/источники, semantic FAIL и постоянные auth/config ошибки не повторяются. First-attempt request identity, P1 и retrieval не изменились.

Публичный комплект содержит только воспроизводимые тесты `test_submission_retry.py`, `test_submission_luna.py`, `test_stronger_submission.py`, `test_public_submission.py`. Полная историческая матрица требует research checkout и не включена. Owner bundle с полными runs/evaluator не предназначен для публикации. Git-репозиторий следует готовить из отдельной публичной release-папки.
