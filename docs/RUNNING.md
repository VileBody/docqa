# Запуск DocQA

[Документация](README.md) · [История решений](EXPERIMENTS.md) · [Главная проекта](../README.md)

Основной сценарий запускает **настоящий сервис с внешними моделями**. Для знакомства без ключей есть отдельный [CPU-demo](#cpu-demo).

## Что понадобится

Docker Engine/Desktop с Compose v2, Python 3.12 для клиентских команд и доступ к модельным endpoints. Команды ниже выполняются в Bash из корня репозитория.

Compose запускает API, Celery, Redis и Qdrant, **но не арендует GPU и не разворачивает модели**. Нужны три функции:

| Функция | Настройка в `.env` |
|---|---|
| Qwen3-Embedding-0.6B, 1024 измерения | `DOCQA_EMBED_BASE_URL`, `DOCQA_EMBED_API_KEY` |
| Qwen3-Reranker-0.6B | `DOCQA_RERANK_BASE_URL`, `DOCQA_RERANK_API_KEY` |
| Sol через Sosana | `DOCQA_OPENAI_BASE_URL`, `DOCQA_OPENAI_API_KEY` |

`OPENAI` — историческое имя слота адаптера, не аттестация провайдера. Revisions и контракт reranker должны совпадать с профилем. Произвольный endpoint с похожим именем может быть несовместим.

Совместимый временный worker описан в [MODEL_ENDPOINTS](MODEL_ENDPOINTS.md). Старые URL удалённых Pods не работают. GPU на клиентском компьютере не нужна при удалённом inference. Проверенная worker-упаковка дополнительно загружает Qwen4B, хотя reader сервиса — Sol; это известный overhead.

## 1. Подготовить копию

```bash
git clone https://github.com/VileBody/docqa.git
cd docqa
python3 -m venv .venv
.venv/bin/pip install -c requirements-stage2-tested.txt '.[rag,service,dev]'
cp .env.example .env
```

Заполните `DOCQA_API_KEY` своим ключом доступа к API (не менее 16 символов) и адреса/ключи моделей. Сохраните основной `sol_p1_runtime.json`, reader-профиль `sol_p1_v2.json`, `DOCQA_TEST_MODE=0`. Пока оставьте `DOCQA_MODEL_CALLS_ENABLED=0`.

Не публикуйте `.env` и `.worker.env`. Если операторский запуск создал `.worker.env`, он содержит временные адреса и секреты worker. Следующая функция сохраняет эти overrides во всех командах:

```bash
dc() {
  if [ -f .worker.env ]; then
    docker compose --env-file .env --env-file .worker.env \
      -f profiles/stage_2/compose.yaml "$@"
  else
    docker compose --env-file .env \
      -f profiles/stage_2/compose.yaml "$@"
  fi
}
```

## 2. Запустить и явно включить модельные вызовы

```bash
dc up --build -d
curl http://127.0.0.1:18080/health
```

`/health` подтверждает процесс и режим, **не готовность внешних моделей**. Проверьте worker health по инструкции подключения.

Модельные обращения ограничивает ledger — файл бюджета, числа попыток и времени. Пример создания:

```bash
# Пример параметров: $2.50, до 256 попыток, 10 800 секунд.
# Владелец запуска самостоятельно задаёт и разрешает свой бюджет.
dc exec -T api python -c \
  'from docqa_rag.budget import Budget; Budget("/data/authorized-budget.json",2.5,256,10800)'
```

Не обнуляйте существующий ledger для обхода лимита. Для новой сессии нужны новый путь и соответствующий бюджет. Аренда GPU и API должны укладываться в один выбранный общий лимит, а не каждый расходовать его целиком.

В `.env` установите `DOCQA_MODEL_CALLS_ENABLED=1` и примените конфигурацию:

```bash
dc up -d
```

Тарифы `profiles/submission/sol_prices.json` исторические. Перед новыми расходами их нужно подтвердить.

## 3. Загрузить документ

В shell укажите тот же ключ, что в `.env`:

```bash
export DOCQA_API_KEY='<ваш ключ DocQA>'

curl -H "Authorization: Bearer $DOCQA_API_KEY" \
  -F classification=synthetic \
  -F file=@examples/acceptance/alpha.txt \
  http://127.0.0.1:18080/documents
```

API принимает только `classification=synthetic` (искусственный учебный текст) или `classification=public` (публичный источник). Метки `private` нет: не помечайте конфиденциальный документ как публичный ради загрузки.

Из ответа возьмите ID задачи и документа. Замените placeholders `TASK_ID` и `DOCUMENT_ID`:

```bash
curl -H "Authorization: Bearer $DOCQA_API_KEY" \
  http://127.0.0.1:18080/tasks/TASK_ID

curl -H "Authorization: Bearer $DOCQA_API_KEY" \
  http://127.0.0.1:18080/documents
```

Дождитесь `ready`. Ответ на загрузку не означает завершения индексации.

## 4. Задать вопрос

```bash
curl -H "Authorization: Bearer $DOCQA_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"document_id":"DOCUMENT_ID","question":"Какой код используется для отслеживания?","diagnostics":true}' \
  http://127.0.0.1:18080/questions
```

Второй пример — «Какова стоимость страхования коробки при доставке?». В опубликованном прогоне первый вопрос дал ALPHA-77 с цитатой, второй — `found=false`. Это исторические результаты, не обещание дословного совпадения будущей формулировки. [API_EXAMPLES](API_EXAMPLES.md).

| Операция | Маршрут |
|---|---|
| Загрузка | `POST /documents` |
| Состояние задачи | `GET /tasks/{id}` |
| Список документов | `GET /documents` |
| Вопрос | `POST /questions` |

Клиент upload/poll/list/QA — [`scripts/submission_http.py`](../scripts/submission_http.py). Его live-режим требует явного authorization согласно контракту скрипта; наличие ключа не запускает его автоматически.

## Ответы, ошибки и диагностика

`complete` означает ответ на все запрошенные части. `partial` — ответ на самостоятельную часть составного вопроса; свободная проза из `missing` не публикуется. `not_found`, `ambiguous`, `contradictory` в текущем профиле дают `found=false` и пустые citations. Прямой запрет и ноль — ответимые случаи. P1 допускает одну простую арифметическую операцию по правилу источника и явно заданным операндам.

| HTTP | Ситуация |
|---|---|
| 404 | Неизвестный документ/задача |
| 409 | Документ ещё не опубликован |
| 422 | Неверный input или несовместимый профиль |
| 429 | Лимит |
| 502 | Невалидный модельный ответ |
| 503 | Модель, источник или хранилище недоступны |

Технический сбой не выдаётся за отсутствие информации. Sol v2 делает максимум **три попытки всего** при классифицированных временных ошибках: reader timeout — до 180 секунд на попытку, embeddings/reranker — до 30 секунд; общий deadline/budget/backoff. Невалидные схема/цитата и смысловая ошибка не повторяются до красивого результата. Неопределённые writes не повторяются вслепую. `POST /tasks/{id}/retry` отдельно повторяет неудавшуюся индексацию.

При `diagnostics=true` ветви dense/BM25 повторно измеряются на том же snapshot и тех же векторах. Это observed replay counts, **не внутренние счётчики первого fusion**. Основные кандидаты берутся из одного native fusion; при `false` диагностические запросы отключены.

## Конфигурация

JSON profile → явно заданные env overrides. API и worker получают одинаковые настройки. Несовместимый override закреплённого reader отклоняется.

| Переменная | Назначение / значение поставки |
|---|---|
| `DOCQA_HTTP_PORT` | Порт API: 18080 |
| `DOCQA_PROFILE` | Общий runtime-профиль |
| `DOCQA_READER_PROFILE` | Sol/P1 v2 и строгий renderer |
| `DOCQA_RERANK_INPUT_LIMIT` | До 24 кандидатов |
| `DOCQA_RERANK_OUTPUT_LIMIT` | До 6 фрагментов |
| `DOCQA_PACK_CHARS` | До 8 000 символов контекста |
| `DOCQA_CHUNK_SIZE`, `DOCQA_CHUNK_OVERLAP` | 1 200 / 160 |
| `DOCQA_FUSION` | `rrf` |
| `DOCQA_BUDGET_LEDGER` | Контроль модельных расходов/попыток |
| `DOCQA_MODEL_CALLS_ENABLED` | Включение платного inference |
| `DOCQA_TEST_MODE` | Явные model doubles |

Полный перечень — [`.env.example`](../.env.example). Sol запрашивается с medium reasoning и 8 192 completion tokens без temperature. В длину входа входят P1, примеры, schema и резерв ответа; внешний tokenizer оценивается консервативно, это не billed tokens.

## CPU-demo

**Проверяет обвязку, не качество AI.** Реальные API/Celery/Redis/Qdrant работают с детерминированными заменами моделей. Нужны Docker и установленные Python dependencies из шага 1; внешние API-ключи и бюджет не требуются.

Demo использует отдельные Compose project, namespace, volumes и порт `18081`. Локальный ключ DocQA создаётся автоматически. Команды выполняйте в Bash из корня репозитория:

```bash
.venv/bin/python - <<'PYDEMO'
from pathlib import Path
import os, secrets
path = Path('.env.demo')
# Существующую конфигурацию не перезаписываем.
values = Path('.env.example').read_text().splitlines()
overrides = {
    'DOCQA_API_KEY': secrets.token_urlsafe(24),
    'DOCQA_TEST_MODE': '1',
    'DOCQA_MODEL_CALLS_ENABLED': '0',
    'DOCQA_HTTP_PORT': '18081',
    'DOCQA_NAMESPACE': 'docqa_demo',
}
for i, line in enumerate(values):
    key = line.split('=', 1)[0]
    if key in overrides:
        values[i] = key + '=' + overrides[key]
fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(fd, 'w') as f:
    f.write('\n'.join(values) + '\n')
PYDEMO

dc_demo() {
  docker compose -p docqa-demo --env-file .env.demo \
    -f profiles/stage_2/compose.yaml "$@"
}
dc_demo up --build -d
curl --retry 30 --retry-all-errors --retry-delay 1 --max-time 3 \
  --retry-max-time 120 --fail http://127.0.0.1:18081/health
export DOCQA_API_KEY="$(.venv/bin/python -c 'from dotenv import dotenv_values; print(dotenv_values(".env.demo")["DOCQA_API_KEY"])')"
.venv/bin/python scripts/submission_http.py \
  --base-url http://127.0.0.1:18081 --output local_cpu_receipt
```

В health должны быть `test_mode=true` и `models_enabled=false`. Если `.env.demo` уже существует, используйте её повторно; для нового запуска клиента выбирайте новую папку результата. Переменные `DOCQA_*`, заданные ранее через `export`, имеют приоритет над env-файлом Compose: используйте чистый терминал или уберите старые overrides.

После проверки:

```bash
dc_demo down
unset DOCQA_API_KEY
```

Данные demo остаются в его отдельных volumes. `.env.demo` игнорируется Git. Такой запуск не подтверждает доступность внешних моделей и не измеряет качество ответов.

## Тесты и завершение

```bash
.venv/bin/python -m pytest tests
```

Публичный набор из 42 тестов самодостаточен: внешние модели заменены mock transport, реальные API-ключи и отдельный GPU не нужны. В полном исследовательском checkout тестов больше, но он не входит в эту поставку. Первый запуск может загружать tokenizer asset; для проверки без сети заранее подготовьте `TIKTOKEN_CACHE_DIR`.

Штатный экспорт готового индекса:

```bash
dc exec -T api python -m docqa_service export /data/snapshot
```

Выберите новый destination. Скопируйте экспорт за пределы временного volume и проверьте восстановление до удаления исходных данных. Restore требует пустого namespace и совместимого fingerprint. Экспорт не является резервной копией всей очереди pending/failed.

```bash
dc down
```

Без `-v` data volumes сохраняются. Команда **не удаляет внешнюю GPU**. Временный RunPod завершите штатным lifecycle и проверьте inventory; инструкция — [MODEL_ENDPOINTS](MODEL_ENDPOINTS.md).

## Данные и эксплуатация

Сервис принимает объявленные публичные или синтетические источники. TXT — источник истины: до 100 логических страниц через `\f`, до 30 MiB; offsets — Unicode-символы неизменённого текста. BOM/NUL/невалидный UTF-8 отклоняются; пустые страницы сохраняются. PDF-адаптер необязателен и выключен по умолчанию; OCR не запускается.

До обработки непубличных данных проверьте допустимость передачи всем внешним моделям. Debug содержит текст, хранится с ограниченными правами и сроком; `docqa_rag.debug.prune_expired` предусмотрен, автоматического планировщика очистки нет.

Для production нужны tenant isolation, TLS/network policy, мониторинг, retention/GC и restore drills. Старые/orphan generations могут занимать диск. Проверки не дают универсальной гарантии смысла ответов.

Команды перенесены из опубликованного README и действующих контрактов. Новое оформление документации не является новым live-прогоном. [Проверенные версии и ограничения](FINAL_REPORT.md).
