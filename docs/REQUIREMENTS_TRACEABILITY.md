# Что реализовано и чем проверено

[Документация](README.md) · [Итоговый отчёт](FINAL_REPORT.md) · [Ответы API](API_EXAMPLES.md)

Матрица связывает требования с реализацией. **PASS означает прохождение указанной проверки в указанной версии**, а не независимую гарантию корректности всех будущих ответов.

Точный прежний реестр с версиями и ссылками на свидетельства сохранён в [REQUIREMENTS_TRACEABILITY.json](REQUIREMENTS_TRACEABILITY.json). Ниже — его читательское представление. Исторические проверки не объявлены повторёнными после смены reader.

## API и работа с документами

| ID | Требование | Подтверждение / ограничение |
|---|---|---|
| T01 | Backend API на FastAPI, без обязательного UI | PASS: FastAPI routes и request/response models |
| T02 | Загрузка документа и запуск индексации | PASS: задача очереди с устойчивым ID |
| T03 | Получение состояния индексации | PASS: ready/failed, 404 для неизвестной задачи |
| T04 | Список загруженных документов | PASS: pending/failed/ready и идемпотентность |
| T05 | Вопрос по конкретному документу | PASS: фильтры документа, snapshot и ready; проверка принадлежности цитат |
| T06 | До 100 страниц; индексация может длиться минуты | PASS: историческая холодная индексация TXT и ответ по странице 100; небольшая fixture, не нагрузочный тест |
| T09 | Независимость вопросов, без сессий/тредов/памяти диалога | PASS: полные inputs H11/H12 совпали; 14 сообщений включают P1 и 6 фиксированных teaching-пар |
| T10 | Отдельная фоновая Celery-задача; брокер Redis | PASS: исторические Celery/Redis и worker SIGKILL |

Подробности инфраструктуры — [исторические свидетельства](submission/evidence/HISTORICAL_INFRASTRUCTURE.json). В финальном Sol-прогоне 100-страничный документ восстанавливали, а не индексировали холодно повторно.

## Хранилище, поиск и контекст

| ID | Требование | Подтверждение / ограничение |
|---|---|---|
| T11 | Qdrant; общая коллекция; фильтр по doc_id | PASS: общая generation collection и document filter |
| T12 | Metadata документа в payload; task state в Redis | PASS: Qdrant payload и Redis task state |
| T13 | Все обязательные payload-поля | PASS: offsets проверяются при публикации и restore |
| T14 | Payload-индексы для полей фильтрации | PASS: payload indexes для doc_id/snapshot/ready |
| T15 | Два вектора в одной точке | PASS: dense и BM25 в каждой точке |
| T16 | Разреженные векторы именно BM25 | PASS: зафиксированная статистика; перестройка словаря при добавлении |
| T17 | Dense/BM25 и объединение внутри одного запроса Qdrant | PASS: один основной native fusion; диагностика отдельно |
| T18 | Способ объединения и лимиты ветвей конфигурируются | PASS: валидация env fusion/input limits |
| T19 | Отдельная модель-reranker после fusion | PASS: исторические 12 вызовов, инструкция worker и trace кандидатов/pack |
| T20 | В генератор идут только верхние фрагменты после reranking | PASS: аутентичные источники и ограниченный top-k |
| T21 | Модель reranker, входной/выходной лимиты — env | PASS: model/revision/input/output limits подключены |
| T26 | Необязательная диагностика ответа | PASS: отдельный replay dense/BM25 на том же snapshot; это не внутренние счётчики первого fusion |

[Как устроен поиск](PROJECT_GUIDE.md) · [Почему оставили текущую конфигурацию](EXPERIMENTS.md).

### Полнота payload — T13

В [`Chunk`](../docqa_rag/types.py) определены все требуемые поля. [`split_document`](../docqa_rag/extraction.py) строит их из неизменного текста; [`Index.build` и `rebuild_snapshot`](../docqa_rag/store.py) записывают `Chunk.model_dump()` в payload одной точки вместе с dense/BM25.

| Поля | Откуда берутся / ограничение |
|---|---|
| `doc_id`, `doc_title` | ID документа и название из upload manifest |
| `chunk_index` | Порядок фрагмента при разбиении |
| `page_from`, `page_to` | Физически один chunk не пересекает логическую TXT-страницу |
| `section` | Поле присутствует; текущий TXT splitter оставляет `null`, структурные разделы не извлекает |
| `text` | Точный диапазон canonical TXT |
| `char_start`, `char_end` | Смещения Unicode-символов; проверяется совпадение с исходной подстрокой |

Это сверка текущего кода и исторических доказательств, не новый live-запуск Qdrant. Интерпретация одной общей коллекции относится к активной generation; старые физические коллекции могут сохраняться. Согласование такой интерпретации с заказчиком не подтверждено.

## Ответ, источники и ошибки

| ID | Требование | Подтверждение / ограничение |
|---|---|---|
| T07 | Ответы только по документу; каждое утверждение имеет основания | PASS на Sol v1 HTTP: 12/12; H03 — not_found, перефразировка и эквивалентность сохранены |
| T08 | При отсутствии сведений: found=false, citations=[], понятное answer | PASS на исторических кейсах и CPU-контрактах; v2 не повторяет смысловые ошибки ради удачного ответа |
| T22 | Splitter, embeddings и LLM через LangChain | PASS: 12 LangChain completions Sol v1, 5 восстановленных документов + 2 учебных TXT; первая попытка v2 локально совпадает с v1 |
| T23 | Валидация структур Pydantic | PASS: Pydantic-контракты и отклонение невалидных model/evidence outputs |
| T24 | Модели/подключения через env | PASS: defaults, env overrides и пример настроек |
| T25 | Обработка ошибок и retries при недоступности LLM/Qdrant | PASS: до 3 попыток всего; временные ошибки, бюджет/deadline, остановка на постоянных/невалидных результатах. CPU MockTransport, не live fault-test |
| T29 | Запросы к API и результаты: точное обозначение и отсутствующий ответ | PASS: точный идентификатор и отказ при непустом pack опубликованы; старый Luna H03 failure учтён отдельно |

[Сохранённые ответы Sol](submission/evidence/SYNTHETIC_HTTP_OUTPUTS.json) · [Изменения v2](submission/evidence/SUBMISSION_V2_CHANGES.json). В старом машинном реестре упоминание H03 failure относится к Luna; текущий публичный Sol H03 — корректный отказ.

### Путь LangChain — T22

| Требуемая часть | Фактический вызов в коде | Уровень проверки |
|---|---|---|
| Splitter | `split_document` → `RecursiveCharacterTextSplitter.create_documents` в [extraction.py](../docqa_rag/extraction.py) | Сверены вызов и восстановление offsets, не только import |
| Embeddings | Фабрика service → `QwenEmbeddings` → `OpenAIEmbeddings.embed_documents/embed_query` в [adapters.py](../docqa_rag/adapters.py) | Прослежен путь от build/query до клиентского адаптера |
| LLM | Фабрика выбранного reader → `ChatAdapter` → `ChatOpenAI.invoke` в [adapters.py](../docqa_rag/adapters.py) | Исторические 12 Sol v1 completions; первая попытка v2 проверена mock transport |

Историческое название `OpenAIEmbeddings`/`ChatOpenAI` — интерфейс LangChain, не подтверждение провайдера. [Фабрики моделей](../docqa_service/models.py) и [фабрика submission](../docqa_service/submission.py) связывают адаптеры с сервисом.

Для T25 отдельно действует [`ReadRetryClient`](../docqa_service/qdrant.py): ограниченные повторы `query_points`, `get_collection`, `scroll`. Неоднозначные writes Qdrant вслепую не повторяются. Документный фильтр T05/T11 ограничивает поиск, но не реализует tenant-авторизацию.

## Поставка и документация

| ID | Требование | Подтверждение / ограничение |
|---|---|---|
| T27 | Исходники и README с инструкцией запуска | PASS: allowlist ZIP, hashes/CRC, проверка доступных секретов; 42/42 public tests. Прежние cleanroom-наборы Luna 30 и Sol v1 14 учитываются отдельно |
| T28 | Архитектура, решения и production-доработки | PASS: устройство, запуск, история выбора и production gaps описаны |

[Проверки упаковки](submission/evidence/SUBMISSION_V2_VALIDATION.json) · [Операционные ограничения](RUNNING.md#данные-и-эксплуатация).
