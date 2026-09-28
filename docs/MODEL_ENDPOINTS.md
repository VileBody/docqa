# Подключение моделей

[Документация](README.md) · [Пошаговый запуск](RUNNING.md) · [Устройство](PROJECT_GUIDE.md)

## Какие подключения нужны

Sosana/Sol — внешний совместимый Chat Completions endpoint. Его base URL и ключ задаёт владелец аккаунта в `DOCQA_OPENAI_BASE_URL` и `DOCQA_OPENAI_API_KEY`. Исторически использовался `https://api.sosana.art/api`; это адрес провайдера, не подтверждение upstream. Тариф перед финальной серией проверен на [странице провайдера](https://sosana.art/): $0.60/$3.00 за миллион входных/выходных токенов, 2026-09-28.

Embeddings/reranker требуют живого совместимого worker. Сохранённые URL уже удалённых Pods повторно использовать нельзя. Можно подключить свои endpoints с закреплёнными моделями и проверенным health либо запустить имеющийся worker через штатный lifecycle. Ни Docker Compose, ни импорт Python не создают аренду.

## Временный worker на RunPod

Ниже пример для оператора **после отдельного разрешения расходов**. Он использует тот же `Session.up/readiness/down`, что прошёл финальную серию; availability и фактический тариф зависят от момента запуска. Выберите доступные `GPU_ID` и `DATA_CENTER_ID` в RunPod. Модельный worker проверяет deadline, idle и потерю heartbeat. Оставьте этот терминал открытым; Enter завершит аренду.

```sh
# Установите RUNPOD_API_KEY, GPU_ID и DATA_CENTER_ID в окружении.
# Сначала установите Python dependencies по docs/RUNNING.md.
.venv/bin/python - <<'PY'
import os, pathlib
from docqa_rag.budget import Budget
from docqa_rag.lifecycle import Session
from docqa_rag.runpod_v2 import RunPodV2API
from docqa_rag.types import Profile
from docqa_rag.util import read
p = Profile.model_validate(read('profiles/submission/sol_p1_runtime.json'))
state = pathlib.Path('.operator/new-session')
# Используйте новую директорию для новой сессии, старую не обнуляйте.
if state.exists():
    raise SystemExit('Choose a new session directory; preserve previous ledger')
budget = Budget(state/'ledger.json', 2.5, 64, 7500)
budget.reserve('storage_upper_bound', .1)
s = Session(state/'session.json', RunPodV2API(os.environ['RUNPOD_API_KEY']), budget,
            duration_s=7200, idle_s=600, hourly_ceiling=1.2,
            tail_loss_accepted=True, budget_derived_duration=True)
bindings = {role:{'model_id':b.model_id,'revision':b.revision}
            for role,b in [('embedding',p.embedding),('reranker',p.reranker),
                           ('generator',p.generators['qwen'])]}
try:
    s.up(image='runpod/pytorch@sha256:cb154fcca15d1d6ce858cfa672b76505e30861ef981d28ec94bd44168767d853',
         gpu_type=os.environ['GPU_ID'], data_center_ids=[os.environ['DATA_CENTER_ID']],
         cloud_type='SECURE', bindings=bindings, worker_file=pathlib.Path('scripts/qwen_worker_v2.py'))
    health=s.readiness()
    assert health['models']==bindings
    assert health['rerank_instruction_contract']=='explicit-instruction-v2'
    # Файл содержит секрет; не печатайте/не публикуйте его.
    path=pathlib.Path('.worker.env')
    fd=os.open(path, os.O_WRONLY|os.O_CREAT|os.O_EXCL, 0o600)
    with os.fdopen(fd,'w') as f:
        for role in ('EMBED','RERANK','QWEN'):
            f.write('DOCQA_'+role+'_BASE_URL='+s.state['endpoint']+'/v1\n')
            f.write('DOCQA_'+role+'_API_KEY='+s.token+'\n')
    print('Worker ready; protected .worker.env written. Keep this process running.')
    input('After exporting/verifying API results, press Enter to release owned Pod: ')
finally:
    s.down(emergency=True)
    print('Owned lifecycle state:',s.state['status'])
PY
```

## Подключение worker к API

Во втором терминале:

```sh
# .env содержит API credential и Sosana; .worker.env — только временный worker.
docker compose --env-file .env --env-file .worker.env \
  -f profiles/stage_2/compose.yaml up --build -d
```

Не изменяйте выбранные модели/revisions между индексацией и QA. После новой аренды меняется URL, поэтому обновите `.worker.env` через новую сессию и пересоздайте API/worker. При shutdown сначала экспортируйте индекс и проверьте restore; затем остановите Compose и завершите терминал lifecycle. `status=deleted` означает подтверждённое удаление собственного Pod; при `cleanup_blocked` выполните выданную штатным lifecycle ручную команду и проверьте inventory. Чужие ресурсы не удалять.

## Расходы и завершение

Бюджет примера worker ($2.5) и ledger API должны укладываться в **один общий разрешённый лимит**. При общем $5 оставьте для API не более $2.5; не создавайте два независимых лимита по $5. Эти суммы и команда — инструкция для будущего оператора, а не продление разрешения завершённого smoke. Веса в release не входят; холодная загрузка требует сети и времени. Проверенная упаковка загружает также Qwen4B, хотя reader здесь Sol.
