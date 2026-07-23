import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
import seaborn as sns


class DotPlotVisualizer:
    def __init__(self, field_color="#e8f0ff", palette="flare_r"):
        self.field_color = field_color
        self.palette = palette

    def _style_axes(self, ax, spine=True):
        ax.set_facecolor(self.field_color)
        ax.patch.set_facecolor(self.field_color)
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(spine)

    def draw_group(self, df, x_col, y_col, hue_col, size_col, title="", highlight_rules=None):
        """
        Полностью автономный метод отрисовки одной группы.
        """
        n_x = df[x_col].nunique()
        n_y = df[y_col].nunique()

        # 1. Динамическая геометрия (из монолитного кода)
        fig_width = 1.4 * n_x + 5.5
        fig_height = 0.5 * n_y + 3.5

        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        fig.patch.set_facecolor(self.field_color)

        # 2. Отрисовка базового dotplot
        sns.scatterplot(
            data=df, x=x_col, y=y_col, hue=hue_col, s=df[size_col],
            palette=self.palette, edgecolor="black", linewidth=0.3, legend=False, ax=ax
        )
        self._style_axes(ax, spine=True)
        ax.set_title(title, fontsize=12, pad=15, weight="bold")

        # 3. ПЕРЕНОС СТРОК (Критично: делается ДО сбора координат)
        plt.xticks(rotation=45, ha="right")
        ax.set_xticklabels([t.get_text().replace(" ", "\n") for t in ax.get_xticklabels()])
        ax.set_yticklabels([t.get_text().replace(" ", "\n") for t in ax.get_yticklabels()])

        # 4. Сдвиг точек влево
        scatter = ax.collections
        if scatter:
            offsets = scatter[0].get_offsets()
            offsets[:, 0] -= 0.15
            scatter[0].set_offsets(offsets)

        # 5. Сбор ЛОКАЛЬНЫХ координат делений
        x_labels = [t.get_text() for t in ax.get_xticklabels()]
        y_labels = [t.get_text() for t in ax.get_yticklabels()]
        x_pos = {label: pos for pos, label in enumerate(x_labels)}
        y_pos = {label: pos for pos, label in enumerate(y_labels)}

        # 6. Отрисовка текста expr и pct внутри точек
        for _, row in df.iterrows():
            x_clean = str(row[x_col]).replace(" ", "\n")
            y_clean = str(row[y_col]).replace(" ", "\n")

            if x_clean not in x_pos or y_clean not in y_pos:
                continue

            base_x = x_pos[x_clean]
            base_y = y_pos[y_clean]

            # Точный расчет сдвига текста (+0.35 от новой позиции точки)
            text_x = base_x - 0.15 + 0.35

            ax.text(text_x, base_y - 0.12, f"{row[hue_col]:.2f}",
                    ha="center", va="center", fontsize=5, color="#5B0600", fontweight="bold")

            # Вывод процентов
            pct_val = row["pct"] if "pct" in df.columns else row[size_col]
            ax.text(text_x, base_y + 0.12, f"{pct_val:.1f}%",
                    ha="center", va="center", fontsize=5, color="#34003C")

        # Настройка внутренних полей холста
        plt.subplots_adjust(left=0.15, right=0.80, top=0.88, bottom=0.18)

        # 7. ЛОКАЛЬНАЯ отрисовка рамок Rectangle
        if highlight_rules:
            for item_name, style in highlight_rules.items():
                item_clean = item_name.replace(" ", "\n")
                if item_clean in x_pos:
                    x_index = x_pos[item_clean]
                    ax.add_patch(Rectangle(
                        (x_index - 0.5, -0.5), 1, n_y,
                        fill=False, zorder=3, **style
                    ))

        return fig, ax

    def add_standard_legends(self, ax, global_df, hue_col, size_min, size_max, alpha, border_legend_items=None):
        """
        Добавление кастомных легенд с учетом новых отступов (bbox_to_anchor=1.25).
        """
        # Цветовая шкала
        norm_c = plt.Normalize(global_df[hue_col].min(), global_df[hue_col].max())
        sm = plt.cm.ScalarMappable(cmap=self.palette, norm=norm_c)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, pad=0.08, shrink=0.7)
        cbar.set_label("Mean expression", fontsize=9)
        cbar.ax.set_facecolor(self.field_color)

        # Легенда размеров кружков
        pct_values = [25, 50, 75, 100]
        size_values = [size_min + ((v / 100) ** alpha) * (size_max - size_min) for v in pct_values]
        handles_sizes = [plt.scatter([], [], s=s, color="gray", edgecolor="black", linewidth=0.5, label=f"{v}%") for
                         s, v in zip(size_values, pct_values)]

        size_legend = ax.legend(
            handles=handles_sizes, title="Percent expressed",
            bbox_to_anchor=(1.25, 0.7), loc="center left", frameon=False,
            borderpad=1.2, labelspacing=1.4, handletextpad=1.2, handlelength=1.8
        )
        ax.add_artist(size_legend)

        # Легенда рамок подсветки
        if border_legend_items:
            border_legend = [Patch(facecolor='none', **item) for item in border_legend_items]
            ax.legend(
                handles=border_legend, title="Highlighted genes",
                bbox_to_anchor=(1.25, 0.15), loc="lower left", frameon=False
            )
