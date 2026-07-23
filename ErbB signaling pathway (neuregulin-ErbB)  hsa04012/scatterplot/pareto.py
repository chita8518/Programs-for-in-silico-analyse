import pandas as pd
import numpy as np
from pathlib import Path

# ============================
# 1. Пути
# ============================

BASE_DIR = Path(__file__).resolve().parent
INPUT_CSV = BASE_DIR / "tau_mean_pct_table.csv"

DOMINANT_DIR = BASE_DIR / "dominant_groups"
DOMINANT_DIR.mkdir(exist_ok=True)

PARETO_DIR = BASE_DIR / "pareto_results"
PARETO_DIR.mkdir(exist_ok=True)

# ============================
# 2. Функция Парето
# ============================

def pareto_front(df, tau_col="tau", expr_col="expr_score"):
    data = df[[tau_col, expr_col]].values
    n = data.shape[0]
    is_optimal = np.ones(n, dtype=bool)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # j доминирует i
            if (data[j][0] >= data[i][0] and data[j][1] >= data[i][1]) and \
               (data[j][0] > data[i][0] or data[j][1] > data[i][1]):
                is_optimal[i] = False
                break

    return df[is_optimal]

# ============================
# 3. Загрузка таблицы
# ============================

df = pd.read_csv(INPUT_CSV, index_col=0)
df["gene"] = df.index

expected_cols = [c for c in df.columns if c.startswith("expected_")]

# ============================
# 4. Доминирующий тип клетки
# ============================

df["dominant_celltype"] = df[expected_cols].idxmax(axis=1)
df["dominant_celltype"] = df["dominant_celltype"].str.replace("expected_", "", regex=False)

# ============================
# 5. Разбиваем на группы + Парето
# ============================

for celltype in df["dominant_celltype"].unique():
    sub = df[df["dominant_celltype"] == celltype].copy()

    # expr_score для этого типа
    expr_col = f"expected_{celltype}"
    sub["expr_score"] = sub[expr_col]

    # сохраняем группу (как раньше)
    dom_file = DOMINANT_DIR / f"{celltype}.txt"
    with open(dom_file, "w", encoding="utf-8") as f:
        f.write(f"Гены с доминирующим типом клетки: {celltype}\n")
        f.write("=" * 60 + "\n\n")
        for gene in sub["gene"]:
            f.write(gene + "\n")

    # Парето только если ≥ 5 генов
    if len(sub) >= 5:
        pareto = pareto_front(sub, tau_col="tau", expr_col="expr_score")
    else:
        pareto = sub.copy()  # маленькие группы — все гены

    # сохраняем Парето
    out_file = PARETO_DIR / f"{celltype}.txt"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(f"Парето-оптимальные гены для клеточного типа: {celltype}\n")
        f.write("=" * 60 + "\n\n")
        for gene in pareto["gene"]:
            f.write(gene + "\n")

    print(f"[OK] {celltype}: {len(pareto)} генов → {out_file}")
