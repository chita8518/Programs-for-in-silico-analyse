import os
from pathlib import Path
import numpy as np
import pandas as pd
import anndata as ad
import matplotlib.pyplot as plt
import seaborn as sns
from adjustText import adjust_text
import settings.HSAD_check
# from utils.output_saver import  save_combined_metrics
from visualizer.dot_plot_visualizer import DotPlotVisualizer

# ==============================================================================
# 1. НАСТРОЙКИ И ИНИЦИАЛИЗАЦИЯ
# ==============================================================================

BASE_DIR = Path(__file__).resolve().parent if "__file__" in locals() else Path(".")
PATH = BASE_DIR / settings.HSAD_check.SIGNAL_PATHWAY
GENE_FILE = BASE_DIR / settings.HSAD_check.GENE_FILE_NAME
OUTPUT_DIR = BASE_DIR / "scatterplot"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
if not OUTPUT_DIR.exists():
    raise RuntimeError(f"Directory not created: {OUTPUT_DIR}")




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
adata = ad.read_h5ad(r"D:\projects\Diplom\scripts\Global_lognormalised.1.h5ad")

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

tau_scores = {gene: tau_specificity(expr[gene]) for gene in expr.columns}
tau_df = pd.DataFrame.from_dict(tau_scores, orient="index", columns=["tau"])

# === expected_expr = expr * pct ===
pct_fraction = pct / 100.0
expected_expr = expr * pct_fraction  # cell_type × gene

# средняя ожидаемая экспрессия по cell types
expected_expr_mean = expected_expr.mean(axis=0)

# === Собираем таблицу с τ, mean_expr и pct по клеткам ===

# pct уже в формате cell_type × gene


# транспонируем, чтобы было gene × cell_type
expected_expr_T = expected_expr.T

# добавляем префикс, чтобы отличать от обычных значений
expected_expr_T = expected_expr_T.add_prefix("expected_")

# собираем финальную таблицу
final_table = (
    tau_df        
    .join(expected_expr_T)  # ожидаемая экспрессия по cell types
)

# Сохраняем
final_table.to_csv(OUTPUT_DIR / "tau_mean_pct_table.csv")
# final_table.to_excel(OUTPUT_DIR / "tau_mean_pct_table.xlsx")

print("Таблица τ + mean_expr + pct сохранена.")
# === Средняя экспрессия пути по каждому типу клеток ===
pathway_mean_by_celltype = expr.mean(axis=1)  # средняя по генам → для каждого cell type

# === Общая средняя экспрессия пути ===
pathway_global_mean = pathway_mean_by_celltype.mean()

# === Сохраняем в текстовый файл в ту же папку ===
# === Средняя экспрессия пути по каждому типу клеток ===

# === Общая средняя экспрессия пути ===
pathway_global_mean = pathway_mean_by_celltype.mean()

# === Сохраняем глобальную активность ===
with open(OUTPUT_DIR / "pathway_activity_summary.txt", "w") as f:
    f.write(f"{pathway_global_mean:.6f}\n")

# === Сохраняем активность по клеточным типам ===
pathway_mean_by_celltype.to_csv(
    OUTPUT_DIR / "pathway_activity_by_celltype.txt",
    sep="\t"
)

print("Сохранены pathway_activity_summary.txt и pathway_activity_by_celltype.txt")
