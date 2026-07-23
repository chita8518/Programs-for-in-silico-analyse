from pathlib import Path
import re


BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR
OUTPUT_FILE = BASE_DIR/Path("pareto_genes_summary.txt")

def clean_cell_type(filename: str) -> str:
    # убираем расширение и возможные (1), (2)
    name = Path(filename).stem
    name = re.sub(r"\(\d+\)$", "", name).strip()
    return name

def extract_genes(text: str) -> list[str]:
    genes = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("Парето-оптимальные"):
            continue
        if set(line) == {"="}:
            continue
        genes.append(line)
    return genes

blocks = []

for file in sorted(INPUT_DIR.glob("*.txt")):
    if file.name == OUTPUT_FILE.name:
        continue
    if file.name.startswith("Summary Statistics"):
        continue

    cell_type = clean_cell_type(file.name)
    text = file.read_text(encoding="utf-8")
    genes = extract_genes(text)

    if not genes:
        continue

    blocks.append((cell_type, genes))

with OUTPUT_FILE.open("w", encoding="utf-8") as out:
    out.write("Анализ позволил определить у сигнального пути Парето-оптимальные гены по следующим группам:\n")

    for cell_type, genes in blocks:
        out.write(f"- Парето-оптимальные гены для клеточного типа: {cell_type}\n")
        for gene in genes:
            out.write(f"\t- {gene}\n")

print(f"Готово: {OUTPUT_FILE.resolve()}")