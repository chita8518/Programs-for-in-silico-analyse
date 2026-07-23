import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re

import settings.basic_script
from utils.output_saver import save_metrics


def split_camel_once(s):
    parts = re.sub(r'(?<!^)(?=[A-Z])', ' ', s).split()
    if len(parts) == 1:
        return s
    return parts[0] + "\n" + " ".join(parts[1:])




# загрузка данных
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
n_plots = settings.basic_script.NUMBER_OF_PLOTS
PATH = BASE_DIR / settings.basic_script.SIGNAL_PATH_FILENAME
# Папка для сохранения диаграмм
SAVE_DIR = BASE_DIR / "Topological_diagrams"
SAVE_DIR.mkdir(exist_ok=True)

dataframe = pd.read_csv(PATH)
print('Исходная таблица:', dataframe.shape)
print(dataframe.head(), '\n')

# определение метрик

metrics =[
    'BetweennessCentrality',
    'Radiality',
    'Degree',
    'NeighborhoodConnectivity',
    'Stress', 
    'TopologicalCoefficient'
]
# метрики которые нужно нормализовать
to_normalize = [
    'BetweennessCentrality',
    'Degree',
    'NeighborhoodConnectivity',
    'Stress'
]

#подготовка таблицы 
# ставим имя гена как индекс
dataframe = dataframe.set_index("name")
# берем только метрики
metrics_matrix = dataframe[metrics].copy()

print('матрица метрик:', metrics_matrix.shape)

# нормируем метрики

metrics_matrix[to_normalize] = (
    metrics_matrix[to_normalize]
    -metrics_matrix[to_normalize].min())/(metrics_matrix[to_normalize].max()-
                                          metrics_matrix[to_normalize].min())


print('Проверка максимум после нормализации')
print(metrics_matrix.max(),'\n')

# режем по квантилям

thresholds = {}

# параметры отбора
# Q = квантиль
# n = количество метрик, которое надо пройти гену, чтобы попасть в выборку
Q_1group = 0.5
Q_2group = 0.7
Q_3group = 0.6
Q_4group = 0.5
n = 4

for m in metrics:
    match m:
        case 'Degree' | 'NeighborhoodConnectivity':
            thresholds[m] = metrics_matrix[m].quantile(Q_1group) 
        case 'BetweennessCentrality' | 'Stress':
            thresholds[m] = metrics_matrix[m].quantile(Q_2group)
        case 'Radiality':
            thresholds[m] = metrics_matrix[m].quantile(Q_3group)
        case 'TopologicalCoefficient':
            thresholds[m] = metrics_matrix[m].quantile(Q_4group)



# отбор по метрикам

selected_by_metrics = {}

for m in metrics:
    if m == 'TopologicalCoefficient':
        selected_by_metrics[m] = metrics_matrix[metrics_matrix[m] <= thresholds[m]]
    else:
        selected_by_metrics[m] = metrics_matrix[metrics_matrix[m] >= thresholds[m]]



important_genes = pd.Index([])

#считаем, по скольким метрикам ген прошел отбор
counts = pd.Series(0, index=metrics_matrix.index) 

for m, df_m in selected_by_metrics.items():
    counts[df_m.index] += 1

# оставляем гены, прошедшие n и более метрик
important_genes = counts[counts >= n].index

output_path = BASE_DIR / "important_genes.txt"
important_genes.to_series().to_csv(output_path, index=False, header=False)
print(f"Список генов сохранён в {output_path}")

important_matrix = metrics_matrix.loc[important_genes]

print(f'Генов, прошедших порог {n} и более метрик:', len(important_genes))

# диагностика

print('NaN по строкам:', metrics_matrix.isna().any(axis=1).sum())
print('Уникальных имен:', metrics_matrix.index.nunique(), '\n')

# аннотации

annot = dataframe.loc[important_genes, [
    "Column3",   # функциональная аннотация PANTHER
    "Column4",   # семейство PANTHER
    "Column5",   # тип белка / категория
]].copy()

annot = annot.rename(columns={
    "Column3": "PANTHER_function",
    "Column4": "PANTHER_family",
    "Column5": "Protein_class",
})
# ================= Настройки кастомного вывода =================
field_color = "#e8f0ff"

# Сортируем матрицу и аннотации по убыванию суммы метрик гена для красивого отображения
important_matrix = important_matrix.loc[important_genes]
important_matrix["_sort_sum"] = important_matrix.sum(axis=1)
important_matrix = important_matrix.sort_values(by="_sort_sum", ascending=False)
important_matrix = important_matrix.drop(columns=["_sort_sum"])
annot = annot.loc[important_matrix.index]


# Вызов функции для сохранения ВСЕЙ таблицы:
save_metrics(important_matrix, base_path=BASE_DIR)

# Нарезаем упорядоченные гены строго на заданное количество групп из настроек
ordered_genes = important_matrix.index.tolist()
gene_groups = np.array_split(ordered_genes, n_plots)

print(
    f"Гены разделены на {n_plots} групп(ы) на основе настроек. Начинаем поочередную отрисовку..."
)

# ================= Итеративная отрисовка групп генов =================
for group_idx, sub_genes in enumerate(gene_groups):
    if len(sub_genes) == 0:
        continue

    # Вырезаем срезы данных для текущей группы генов
    matrix_chunk = important_matrix.loc[sub_genes]
    annot_chunk = annot.loc[sub_genes]
    current_chunk_size = len(sub_genes)

    # --- 1. Отрисовка таблицы аннотаций для текущей группы ---
    fig_tbl, ax_tbl = plt.subplots(figsize=(12, current_chunk_size * 0.35 + 1))
    ax_tbl.axis("off")

    tbl = ax_tbl.table(
        cellText=annot_chunk.values,
        rowLabels=annot_chunk.index,
        colLabels=annot_chunk.columns,
        loc="center",
        cellLoc="left",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7)
    tbl.scale(1, 1.2)

    ax_tbl.set_title(
        f"Аннотации PANTHER — Группа {group_idx + 1} из {n_plots}",
        fontsize=10,
        pad=10,
        loc="left",
        weight="bold",
    )
    fig_tbl.savefig(
        SAVE_DIR / f"group_{group_idx + 1:02d}_annotation.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.tight_layout()

    # --- 2. Отрисовка хитмапа для текущей группы ---
    fig_hm, ax_hm = plt.subplots(figsize=(10, 0.5 * current_chunk_size + 2))
    sns.set(font_scale=0.9)

    # Возвращаем исходный shrink=0.6 и жестко задаем границы меток шкалы (ticks) от 0 до 1
    sns.heatmap(
        matrix_chunk,
        cmap="flare_r",
        linewidths=0.5,
        linecolor=field_color,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 8},
        cbar_kws={
            "shrink": 0.6,
            "ticks": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        },  # "Единичка" теперь зафиксирована на шкале
        vmin=0.0,  # Гарантируем нижнюю границу цвета
        vmax=1.0,  # Гарантируем верхнюю границу цвета (1.0)
        yticklabels=True,
        xticklabels=True,
        ax=ax_hm,
    )

    # Кастомизация подписей оси X с использованием функции split_camel_once
    plt.xticks(rotation=0)
    labels = [
        split_camel_once(label.get_text()) for label in ax_hm.get_xticklabels()
    ]
    ax_hm.set_xticklabels(labels, ha="center", va="top")

    plt.yticks(rotation=0)

    # Настройка фирменных фонов группы
    ax_hm.set_facecolor(field_color)
    fig_hm.patch.set_facecolor(field_color)

    ax_hm.set_title(
        f"Тепловая карта метрик — Группа {group_idx + 1} из {n_plots}",
        fontsize=11,
        pad=15,
        weight="bold",
    )

    # Корректировка отступов, чтобы длинные названия генов слева не обрезались
    plt.subplots_adjust(left=0.3, right=0.9, top=0.9, bottom=0.2)
    plt.tight_layout()
    fig_hm.savefig(
        SAVE_DIR / f"group_{group_idx + 1:02d}_heatmap.png",
        dpi=300,
        bbox_inches="tight"
    )

print("Рендеринг всех окон визуализации...")
plt.show()