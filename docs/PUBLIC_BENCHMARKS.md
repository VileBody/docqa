# Внешняя проверка: ContractNLI и IIRC

[Документация](README.md) · [История решений](EXPERIMENTS.md) · [Как читать результаты](RESULTS_GUIDE.md)

После внутренних экспериментов проверили прежние зафиксированные профили на двух внешних англоязычных задачах. Это проверка переноса на другой материал. **Она выполнена до выбора Sol/P1 и не измеряет качество текущего reader.**

## Протокол исторического пилота

| Параметр | ContractNLI | IIRC |
|---|---|---|
| Официальная постановка | [Stanford ContractNLI](https://stanfordnlp.github.io/contract-nli/): метка отношения и supporting spans | [Статья IIRC](https://aclanthology.org/2020.emnlp-main.86/): вопрос к неполному начальному контексту и связанным статьям |
| Замороженный evaluation-срез | 20 договоров из test × все 17 гипотез = 340 | 100 вопросов из dev; train использовали только для интерфейсного smoke |
| Отбор | Техническая пригодность, затем сортировка SHA-256 от `seed/benchmark/split/id` | Та же процедура; seed **20260927**, gold не использовали для отбора |
| Вход и область поиска | Гипотеза и текст выбранного договора; оригинальные spans/labels доступны evaluator | Данный seed и **все входные first-hop links** из offline snapshot; без annotator `question_links/context` |
| Что не разрешено | Подмена native labels ответами свободного QA | Произвольный web, second-hop URLs, дополнительные документы или калькулятор |
| Выход / scoring | Нативные метки, отдельная проверка evidence spans | Авторские EM и numeric-aware aligned multi-span F1 |

Adapter — `public-native-tasks-en-v1`, serializer — `html-data-entities-preserved-order-v2`. Изменение serializer выполнено до inference, IDs среза сохранены. По одному логическому TXT-листу на источник, не физическая PDF-пагинация. Ограничения отбора: до 2 MiB на источник, 16 MiB на case bundle, 16 000 UTF-8 bytes seed, 100 input links. Непригодные/отсутствующие статьи исключались с причиной, без замены новым web-текстом.

Flow в IIRC искал по вопросу + первым 512 символам seed + названию корневой статьи (до 2 048 символов); полный seed и links оставались доступны модельным входам. Во всех режимах начальный passage дан до поиска; IDF построен на общей замороженной union-коллекции. Финальный synthesis — fresh `reader_replay`, не продолжение истории эпизода. Flow ограничен одним поиском, static/agent — четырьмя; это сравнение систем с указанными бюджетами, не равная стоимость вызовов.

### Версии scorer и данных

- ContractNLI: [dataset metadata commit](https://github.com/stanfordnlp/contract-nli/tree/eced6528dd3c1d14d73f9a87df8f7bdbc03126f9). Загружены structured JSON из авторского ZIP; hashes отдельных members сохранены, SHA всего ZIP **нет**, поскольку PDF-часть не скачивали.
- ContractNLI: [author evaluation.py](https://github.com/stanfordnlp/contract-nli-bert/blob/058c56fd62d56897bb4fcfbf1be71b17aee3a79c/contract_nli/evaluation.py), функции `evaluate_class` / `evaluate_predicted_spans`. Приведённый ниже macro-F1 по трём классам — наша явная агрегация; это не переименование авторских E/C means. Ranking AP — N/A: бинарные citations не дают calibrated scores всех spans.
- IIRC: [репозиторий автора](https://github.com/jferguson144/IIRC-baseline/tree/fa397b2bbee54c71861abbb7a379d6999552bcac), [drop_eval.py](https://github.com/jferguson144/IIRC-baseline/blob/fa397b2bbee54c71861abbb7a379d6999552bcac/numnet_plus/drop_eval.py). `get_metrics` сохранён с числовым matching, выравниванием нескольких spans и округлением; primary projection следует авторскому `make_drop_style.py`.

Знаменатель основных метрик — весь фиксированный план, ошибки не заменяются корректным `NotMentioned`/отказом. Отдельные official completed-only показатели не подменяют основную оценку. Договоры и общие корневые/linked articles создают зависимость; 340 утверждений и 100 вопросов не являются таким же числом независимых документов.

## ContractNLI: утверждение относительно договора

Нужно классифицировать утверждение: подтверждено, опровергнуто или не упомянуто. Это классификация, а не свободный QA-ответ.

| Reader | Правильный класс | Macro-F1 | Нативные completions |
|---|---:|---:|---:|
| Qwen | 199/340 — **58,53%** | 56,11% | 336 |
| Sosana / Luna | 247/340 — **72,65%** | 67,92% | 338 |

Ошибки исполнения остаются в фиксированном знаменателе. Корректная метка не доказывает корректность каждого пояснения или цитаты. Наблюдаемый выигрыш Luna относится к этому срезу и этим профилям.

## IIRC: начального контекста может не хватать

| Режим, reader Qwen | Exact Match | F1 | Завершённые случаи |
|---|---:|---:|---:|
| Flow | **29%** | **31,70%** | 98/100 |
| Static | 25% | 27,03% | 87/100 |
| Agent | N/A | N/A | 0/100: evaluation заблокирована train-gate |

Exact Match требует совпадения нормализованного ответа; F1 учитывает перекрытие токенов. У заблокированного агента нет измеренной evaluation-точности: N/A нельзя заменить на 0%.

## Как это повлияло на решение

Результаты не дали оснований объявить преимущество агентного пути. Они также показали разницу между небольшой внутренней приёмкой и внешней задачей. Поэтому основной сервис сохранил flow, а общую точность по цифрам авторского контроля не обещает.

Sosana upstream независимо не подтверждён. Прежнее исследование стадии 3 осталось частично завершённым: Qwen-контроллер не прошёл smoke, а все восемь dependency-кейсов допускали достаточный первый flow-контекст. [Отдельное сравнение агентов и authored reserved](EXPERIMENTS.md#3-нужен-ли-агент-с-несколькими-поисками).

## Происхождение

Это переоформление ранее опубликованной сводки с уточнением протокола по сохранённым PLAN/SAMPLING/SOURCES_AND_PROTOCOL_NOTES, без новых прогонов или настройки на evaluation. Полные pilots и evaluator data не входят в открытый репозиторий. В полном research checkout активный указатель — `reports/public_benchmarks/ACTIVE_HANDOFF.json`, каноническая сводка — `reports/final_acceptance/v1/CANONICAL_PUBLIC_RESULTS.json`; исходные отчёты и judgments — в `reports/public_benchmarks/v1`.

Публичных данных достаточно для чтения результатов, но это не полный пакет воспроизведения benchmark и не заявка в leaderboard.

Версии и параметры выше проверены по локальным историческим записям. Полные IDs, exclusions, file manifests, adapter/scorer assets и individual predictions публично не поставляются; самих ссылок на авторов недостаточно для полного воспроизведения нашего пилота.
