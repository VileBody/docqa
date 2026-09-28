"""Generate README visuals from versioned, published summaries.

Run: python scripts/render_readme_visuals.py
Only chart generation needs matplotlib; the DocQA runtime does not.
No network, model calls, secrets, dataset access, or font files.
"""
from __future__ import annotations

import json
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / 'docs' / 'assets'


def text(x, y, value, size=20, weight='400', fill='#253449'):
    return f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" fill="{fill}">{escape(value)}</text>'


def lines(x, y, values, size=19, step=28, fill='#536173'):
    return ''.join(text(x, y + i * step, value, size, fill=fill) for i, value in enumerate(values))


def rect(x, y, w, h, fill='#ffffff', stroke='#d8e1eb', r=16):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}"/>'


def arrow(x1, y1, x2, y2):
    return f'<path d="M{x1},{y1} L{x2},{y2}" stroke="#8392a4" stroke-width="2.4" marker-end="url(#arrow)" fill="none"/>'


def svg(body, width, height, title, description):
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">
<title id="title">{escape(title)}</title><desc id="desc">{escape(description)}</desc>
<defs><marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 Z" fill="#8392a4"/></marker></defs>
<rect width="100%" height="100%" rx="20" fill="#f6f8fb"/>
<g font-family="DejaVu Sans, Arial, sans-serif">{body}</g></svg>'''


def diagrams():
    body = text(34, 43, 'ДОКУМЕНТ → ОСНОВАНИЯ → ОТВЕТ', 17, '700', '#536173')
    cards = [
        (34, '01', 'Ваш документ', ['Текст договора,', 'регламента или инструкции']),
        (391, '02', 'Найденный текст', ['Поиск по смыслу и словам,', 'повторное ранжирование']),
        (748, '03', 'Ответ с источником', ['Исходная цитата,', 'страница и координаты']),
    ]
    for x, number, title, detail in cards:
        body += rect(x, 70, 322, 180)
        body += text(x + 22, 106, number, 18, '700', '#147d86')
        body += text(x + 22, 147, title, 24, '700')
        body += lines(x + 22, 184, detail, 18, 27)
    body += arrow(363, 160, 381, 160) + arrow(720, 160, 738, 160)
    body += text(34, 291, 'Не хватает нужных сведений? Предусмотрен отказ, а не ответ по памяти.', 21)
    (ASSETS / 'docqa-overview.svg').write_text(svg(body, 1104, 325, 'От документа к ответу', 'TXT, два поиска, отбор и привязка ответа к исходному тексту.'), encoding='utf-8')

    body = text(32, 47, 'Один сервис. Два маршрута.', 32, '700')
    body += rect(22, 75, 1060, 240, '#eef3f8', '#eef3f8')
    body += text(43, 109, '01   ПОДГОТОВКА ДОКУМЕНТА — В ФОНЕ', 17, '700', '#536173')
    top = [
        (43, 'Загрузка TXT', ['API принимает файл', 'и возвращает ID задачи']),
        (308, 'Celery', ['Индексация в фоне', 'Redis: очередь, статус']),
        (573, 'Два вектора', ['Qwen: векторы смысла', 'BM25: поиск по словам']),
        (838, 'Qdrant', ['Текст + два вектора', 'Готовый индекс']),
    ]
    for x, title, detail in top:
        body += rect(x, 130, 223, 141)
        body += text(x + 15, 165, title, 23, '700')
        body += lines(x + 15, 204, detail, 16, 26)
    for x in [276, 541, 806]:
        body += arrow(x, 197, x + 23, 197)
    body += text(43, 298, 'До завершения индексации документ недоступен для вопросов.', 19)
    body += rect(22, 340, 1060, 280, '#eef5f5', '#eef5f5')
    body += text(43, 377, '02   ПОИСК И ОТВЕТ — ДЛЯ КАЖДОГО ВОПРОСА', 17, '700', '#536173')
    bottom = [
        (43, 'Вопрос + документ', ['Выбранный документ', 'Без прошлого диалога']),
        (308, 'Поиск в Qdrant', ['Смысл + BM25', 'Объединение выдач']),
        (573, 'Qwen reranker', ['Оценка кандидатов', 'Отбирает контекст']),
        (838, 'Sol + проверка', ['Модель пишет ответ', 'Код проверяет цитаты']),
    ]
    for x, title, detail in bottom:
        body += rect(x, 400, 223, 141)
        body += text(x + 15, 435, title, 20, '700')
        body += lines(x + 15, 474, detail, 16, 26)
    for x in [276, 541, 806]:
        body += arrow(x, 467, x + 23, 467)
    body += lines(43, 577, ['Генератор видит только отобранные источники.', 'На выходе — ответ с цитатами или сообщение о недостатке сведений.'], 19, 28)
    body += text(32, 659, 'Qwen ищет и ранжирует. Sol формулирует. Приложение проверяет происхождение.', 19)
    (ASSETS / 'docqa-pipeline.svg').write_text(svg(body, 1104, 692, 'Архитектура DocQA', 'Фоновая индексация и независимый маршрут поиска, ранжирования и ответа.'), encoding='utf-8')


def chart(labels, values, *, title, subtitle, xlabel, xmax, value_labels, footer, name):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # Default plotting colours, one chart per figure, zero baseline.
    with matplotlib.rc_context({'svg.fonttype': 'none', 'svg.hashsalt': 'docqa-readme-v1'}):
        fig, ax = plt.subplots(figsize=(11.04, 4.1))
        fig.subplots_adjust(left=.20, right=.95, top=.68, bottom=.30)
        ys = list(range(len(values)))
        ax.barh(ys, values, height=.48)
        ax.set_yticks(ys, labels, fontsize=16)
        ax.invert_yaxis()
        ax.set_xlim(0, xmax)
        ax.set_xlabel(xlabel, fontsize=13, labelpad=9)
        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', length=0, pad=12)
        ax.set_axisbelow(True)
        ax.xaxis.grid(True, alpha=.18)
        for side in ('top', 'right', 'left'):
            ax.spines[side].set_visible(False)
        for i, (value, label) in enumerate(zip(values, value_labels)):
            ax.text(value + xmax * .016, i, label, va='center', fontsize=16, fontweight='bold')
        fig.text(.036, .90, title, fontsize=22, fontweight='bold', va='top')
        fig.text(.036, .80, subtitle, fontsize=13.5, va='top')
        fig.text(.036, .045, footer, fontsize=11.5, va='bottom', linespacing=1.4)
        fig.savefig(ASSETS / (name + '.svg'), metadata={'Date': None})
        fig.savefig(ASSETS / (name + '.png'), dpi=150)
        plt.close(fig)


def main():
    ASSETS.mkdir(parents=True, exist_ok=True)
    data = json.loads((ASSETS / 'metrics.json').read_text(encoding='utf-8'))
    r = data['reader']
    # Fail on stale chart data if the corresponding public source exists.
    source = ROOT / r['source']
    if source.exists():
        recorded = json.loads(source.read_text(encoding='utf-8'))['comparison']
        for i, key in enumerate(r['keys']):
            for field in ('median_generation_s', 'list_price_usage_usd'):
                if abs(recorded[key][field] - r[field][i]) > 1e-12:
                    raise ValueError(f'Chart/source mismatch: {key}.{field}')
            if recorded[key]['semantic_pass'] != r['semantic_pass'][i]:
                raise ValueError('Reader success count changed: review the narrative before plotting')
    diagrams()
    chart(r['labels'], r['median_generation_s'],
          title='Одинаковый результат контроля — разная задержка',
          subtitle='Sol и Astra: по 40/40 случаев; медиана новых модельных ответов',
          xlabel='Время генерации, секунды · меньше — быстрее', xmax=6.1,
          value_labels=['3,43 с', '4,94 с'],
          footer='38 новых ответов на модель. Измерен reader, не весь HTTP-путь.\nАвторский контроль с повторами; не публичный benchmark.', name='reader-latency')
    chart(r['labels'], r['list_price_usage_usd'],
          title='Стоимость той же контрольной серии',
          subtitle='Историческая оценка по сохранённым тарифам и usage',
          xlabel='Доллары США · меньше — дешевле', xmax=.38,
          value_labels=['$0,0621', '$0,3080'],
          footer='38 новых ответов на модель. Оценка, не invoice и не текущий прайс.\nКачество на этом контроле: по 40/40 с двумя локальными отказами.', name='reader-cost')
    c = data['contractnli']
    vals = [a / b * 100 for a, b in zip(c['correct'], c['total'])]
    chart(c['labels'], vals,
          title='ContractNLI: историческая внешняя проверка',
          subtitle='Доля правильных классов на срезе: 340 случаев на reader',
          xlabel='Label accuracy, % · выше — больше совпадений с gold', xmax=100,
          value_labels=['58,53%', '72,65%'],
          footer='Прежние модели и профили. Это НЕ результат выбранного Sol/P1.\nКлассификация договора, не общая точность свободных QA-ответов.', name='contractnli-historical')
    print('Created 2 diagrams and 3 charts (SVG; charts also PNG). No inference performed.')


if __name__ == '__main__':
    main()
