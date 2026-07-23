import os
from pathlib import Path
import numpy as np
import pandas as pd
import anndata as ad
import matplotlib.pyplot as plt
import seaborn as sns

import settings.HSAD_check
from utils.output_saver import  save_combined_metrics
from visualizer.dot_plot_visualizer import DotPlotVisualizer

# ==============================================================================
# 1. НАСТРОЙКИ И ИНИЦИАЛИЗАЦИЯ
# ==============================================================================

BASE_DIR = Path(__file__).resolve().parent if "__file__" in locals() else Path(".")
PATH = BASE_DIR / settings.HSAD_check.SIGNAL_PATHWAY
GENE_FILE = BASE_DIR / settings.HSAD_check.GENE_FILE_NAME
OUTPUT_DIR = BASE_DIR / "expression diagrams"
OUTPUT_DIR.mkdir(exist_ok=True)


# Переменные оформления
field_color = "#e8f0ff"
size_min, size_max = 50, 400
alpha = 1
top_n = 3
cardio_cells = ["Atrial Cardiomyocyte", "Ventricular Cardiomyocyte"]


# ==============================================================================
# 2. ФУНКЦИИ РАСЧЕТА ДАННЫХ
# ==============================================================================
def get_expression_by_celltype(adata, gene_list):
    mask = adata.var["gene_name"].isin(gene_list)
    found_genes = adata.var["gene_name"][mask].tolist()
    if len(found_genes) == 0:
        return None
    sub = adata[:, mask]
    df = pd.DataFrame(sub.X.toarray(), columns=found_genes)
    df["cell_type"] = adata.obs["cell_type"].values
    return df.groupby("cell_type").mean()


def get_pct_by_celltype(adata, gene_list):
    mask = adata.var["gene_name"].isin(gene_list)
    found_genes = adata.var["gene_name"][mask].tolist()
    if len(found_genes) == 0:
        return None
    sub = adata[:, mask]
    df = pd.DataFrame(sub.X.toarray(), columns=found_genes)
    df["cell_type"] = adata.obs["cell_type"].values
    return df.groupby("cell_type").apply(lambda x: (x[found_genes] > 0).sum() / len(x) * 100)


def tau_specificity(values):
    if values.max() == 0:
        return 0.0
    x_hat = values / values.max()
    n = len(values)
    return ((1 - x_hat).sum()) / (n - 1)


def style_axes(ax, bg=None, grid=False, spine=False):
    if bg is not None:
        ax.set_facecolor(bg)
        ax.patch.set_facecolor(bg)
    for s in ax.spines.values():
        s.set_visible(spine)
    ax.grid(grid, color="#5e819e", linewidth=1.2)
    if grid:
        ax.set_axisbelow(True)


# ==============================================================================
# 3. ЗАГРУЗКА И ПЕРВИЧНАЯ ОБРАБОТКА ДАННЫХ
# ==============================================================================
adata = ad.read_h5ad(settings.HSAD_check.HHCA_DATASET)

# Сборка единого gene_name
adata.var["gene_name"] = adata.var[
                             ["gene_name_scRNA-0-original",
                              "gene_name_snRNA-1-original",
                              "gene_name_multiome-2-original"]
                         ].bfill(axis=1).iloc[:, 0]

with open(GENE_FILE, "r") as f:
    gene_list = [line.strip() for line in f if line.strip()]

print("Загружено генов:", len(gene_list))

expr = get_expression_by_celltype(adata, gene_list)
pct = get_pct_by_celltype(adata, gene_list)

# Расчет рангов топ-N
expr_rank_df = expr.rank(axis=0, method="dense", ascending=False)
pct_rank_df = pct.rank(axis=0, method="dense", ascending=False)

genes_top_expr = [gene for gene in expr.columns if any(expr_rank_df.loc[cell, gene] <= top_n for cell in cardio_cells)]
genes_top_pct = [gene for gene in pct.columns if any(pct_rank_df.loc[cell, gene] <= top_n for cell in cardio_cells)]

# Сборка long-формата
expr_df = expr.reset_index().melt(id_vars="cell_type", var_name="gene", value_name="expression")
pct_df = pct.reset_index().melt(id_vars="cell_type", var_name="gene", value_name="pct")

plot_df = expr_df.merge(pct_df, on=["cell_type", "gene"])
plot_df["gene"] = plot_df["gene"].astype("category")

plot_df["expr_norm"] = plot_df["expression"] / plot_df["expression"].max()
plot_df["pct_norm"] = plot_df["pct"] / plot_df["pct"].max()
plot_df["size_manual"] = (plot_df["pct"] / 100) ** alpha
plot_df["size_scaled"] = size_min + plot_df["size_manual"] * (size_max - size_min)

# Расчет специфичности tau и аннотаций
tau_scores = {gene: tau_specificity(expr[gene]) for gene in expr.columns}
tau_df = pd.DataFrame.from_dict(tau_scores, orient="index", columns=["tau"])

annot = pd.read_csv(PATH).set_index("name").rename(columns={"Column3": "PANTHER_function", "Column4": "PANTHER_family"})
merged_sorted = tau_df.join(annot, how="left")[["tau", "PANTHER_function"]].sort_values(by="tau", ascending=False)

# ==============================================================================
# 4. ДЕКЛАРАТИВНЫЕ ПРАВИЛА ПОДСВЕТКИ И ВЫЗОВ DOTPLOT ПО ГРУППАМ
# ==============================================================================

# 1. Формируем единый словарь правил подсветки с четкими приоритетами цветов
genes_both = set(genes_top_expr) & set(genes_top_pct)
highlight_rules = {}

for gene in gene_list:
    if gene in genes_both:
        highlight_rules[gene] = {'edgecolor': '#228B22', 'linewidth': 2.5}
    elif gene in genes_top_expr:
        highlight_rules[gene] = {'edgecolor': 'blue', 'linewidth': 2}
    elif gene in genes_top_pct:
        highlight_rules[gene] = {'edgecolor': 'orange', 'linewidth': 2}

# 2. Описываем метаданные для построения легенды рамок выделения
border_meta = [
    {'edgecolor': 'blue', 'linewidth': 2, 'label': f'Top-{top_n} \nmean expression'},
    {'edgecolor': 'orange', 'linewidth': 2, 'label': f'Top-{top_n} \npercent expressed'},
    {'edgecolor': '#228B22', 'linewidth': 2.5, 'label': 'Both criteria'}
]

# 3. Нарезаем упорядоченные по tau гены на заданное количество групп
ordered_genes = merged_sorted.index.tolist()
n_plots = settings.HSAD_check.NUMBER_OF_PLOTS
gene_groups = np.array_split(ordered_genes, n_plots)

# 4. Инициализируем наш абстрактный визуализатор
visualizer = DotPlotVisualizer(field_color=field_color, palette="flare_r")

print(f"Гены разделены на {n_plots} групп(ы). Начинаем поочередную отрисовку...")

save_combined_metrics(plot_df=plot_df, final_genes=ordered_genes, base_path=BASE_DIR)


# 5. Итеративно строим подграфики для каждой группы генов
for group_idx, sub_genes in enumerate(gene_groups):
    if len(sub_genes) == 0:
        continue

    # Вырезаем срез данных для текущей группы и очищаем пустые категории
    sub_plot_df = plot_df[plot_df["gene"].isin(sub_genes)].copy()
    sub_plot_df["gene"] = sub_plot_df["gene"].cat.remove_unused_categories()

    # Делегируем классу построение базовой геометрии, сдвигов и текста
    fig, ax = visualizer.draw_group(
        df=sub_plot_df,
        x_col="gene",
        y_col="cell_type",
        hue_col="expr_norm",
        size_col="size_scaled",
        title=f"Группа генов {group_idx + 1} из {n_plots}",
        highlight_rules=highlight_rules
    )
    fig.savefig(
        OUTPUT_DIR / f"dotplot_group_{group_idx + 1}.png",
        dpi=300,
        bbox_inches="tight"
    )

    # Навешиваем стандартизированный блок легенд со смещением в правый край
    visualizer.add_standard_legends(
        ax=ax,
        global_df=plot_df,
        hue_col="expr_norm",
        size_min=size_min,
        size_max=size_max,
        alpha=alpha,
        border_legend_items=border_meta
    )
def plot_tau_barplot(tau_df, field_color="#e8f0ff"):
    """
    Строит масштабируемый горизонтальный barplot для τ-специфичности.
    tau_df — DataFrame вида: index = gene, column = 'tau'
    """

    # сортировка по убыванию
    tau_sorted = tau_df.sort_values("tau", ascending=False)  # вверх — самые специфичные

    # динамическая высота фигуры
    n_genes = len(tau_sorted)
    fig_width = max(10, n_genes * 0.35)

    fig, ax = plt.subplots(figsize=(fig_width, 6))
    fig.patch.set_facecolor(field_color)
    ax.set_facecolor(field_color)

    sns.barplot(
        data=tau_sorted,
        y=tau_sorted["tau"],
        x=tau_sorted.index,
        palette="flare",
        ax=ax
    )

    ax.set_xlabel("Tau specificity", fontsize=12, weight="bold")
    ax.set_ylabel("Gene", fontsize=12, weight="bold")

    # убираем лишнее
    sns.despine(left=True, bottom=True)

    ax.set_ylabel("Tau specificity", fontsize=12, weight="bold")
    ax.set_xlabel("Gene", fontsize=12, weight="bold")

    # подписи генов под углом
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, fontsize=8)

    for i, (gene, row) in enumerate(tau_sorted.iterrows()):
        tau_val = row["tau"]
        ax.text(
            i, tau_val + 0.01,  # позиция над столбом
            f"{tau_val:.2f}",  # формат: 1 знак после запятой
            ha="center", va="bottom",
            fontsize=8, weight="bold"
        )
    plt.tight_layout()
    out_path = OUTPUT_DIR / "tau_specificity_barplot.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Barplot сохранён в: {out_path}")

plot_tau_barplot(tau_df)

print("Рендеринг всех окон...")
plt.show()
