import csv

from openpyxl.reader.excel import load_workbook
from openpyxl.styles import Alignment

import settings.HSAD_check

import settings.basic_script


def save_metrics(matrix_df, base_path):
    """
    Сохраняет переданную матрицу метрик в CSV файл.
    Индексы (имена генов) сохраняются как первая колонка с названием 'name'.

    :param matrix_df: pandas.DataFrame с метриками генов
    :param file_name: Имя выходного CSV-файла или полный путь к нему
    """
    output_csv =  base_path / settings.basic_script.OUT_CSV_FILENAME

    output_xlsx = base_path / settings.basic_script.OUT_XLSX_FILENAME

    # CSV
    matrix_df.to_csv(output_csv, index=True, index_label="name")

    # Excel
    df_out = matrix_df.reset_index().rename(columns={"index": "name"})
    df_out.to_excel(output_xlsx, index=False)

    # === Автоширина колонок ===
    wb = load_workbook(output_xlsx)
    ws = wb.active

    for column_cells in ws.columns:
        max_length = 0
        column = column_cells[0].column_letter

        for cell in column_cells:
            try:
                cell_value = str(cell.value)
                max_length = max(max_length, len(cell_value))
            except:
                pass

        ws.column_dimensions[column].width = max_length * 1.2

    wb.save(output_xlsx)
    print(f"Данные метрик успешно сохранены в: {output_xlsx}")


def save_combined_metrics(
    plot_df, final_genes, base_path
):
    """Сохраняет матрицу, используя НОРМИРОВАННЫЕ значения expr_norm и pct_norm

    из plot_df в формате "m = x \n p = y".
    """
    # 1. Фильтруем таблицу по нужным генам
    filtered_df = plot_df[plot_df["gene"].isin(final_genes)].copy()

    # 2. Собираем текст: m = нормированная экспрессия, p = реальный процент (pct)
    filtered_df["combined_text"] = filtered_df.apply(
        lambda row: f"m = {row['expr_norm']:.5f} \n p = {row['pct']:.5f}",
        axis=1
    )

    # 3. Разворачиваем в матрицу
    combined_matrix = filtered_df.pivot(index="cell_type", columns="gene", values="combined_text")
    combined_matrix = combined_matrix[final_genes]
    final_table = combined_matrix.T

    # 4. Сохраняем CSV
    output_csv = base_path / settings.HSAD_check.OUT_CSV_FILENAME

    final_table.to_csv(output_csv, index=True, index_label="gene", sep=",", quoting=csv.QUOTE_NONNUMERIC)

    # === 5. Сохраняем Excel ===
    output_xlsx = base_path / settings.HSAD_check.OUT_XLSX_FILENAME
    final_table.to_excel(output_xlsx)

    # === 6. Включаем перенос строк в Excel ===
    wb = load_workbook(output_xlsx)
    ws = wb.active


    for row in ws.iter_rows():
        for cell in row:
            cell.alignment = Alignment(wrap_text=True, vertical="top")

    for column_cells in ws.columns:
        max_length = 0
        column = column_cells[0].column_letter

        for cell in column_cells:
            try:
                cell_value = str(cell.value)
                max_length = max(max_length, len(cell_value))
            except:
                pass

        ws.column_dimensions[column].width = max_length * 1.2

    wb.save(output_xlsx)
    print(f"Таблица успешно сохранена")

