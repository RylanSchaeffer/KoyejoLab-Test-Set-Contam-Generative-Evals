"""Plotting utilities for publication-quality figures.

This module provides helper functions for matplotlib/seaborn visualizations
with LaTeX rendering enabled. Figures are styled for academic publications.

Default settings:
    - LaTeX text rendering with Computer Modern font
    - Seaborn whitegrid style
    - Font size 23pt
"""

import os
from typing import Any, Union

import matplotlib.pyplot as plt
import seaborn as sns


# Configure seaborn style
sns.set_style("whitegrid")

# Enable LaTeX rendering for publication-quality text
# See: https://stackoverflow.com/a/23856968
plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "Computer Modern"
plt.rcParams["text.latex.preamble"] = "\n".join([r"\usepackage{amsmath}"])
plt.rcParams["font.size"] = 23


def format_g_legend_in_scientific_notation(g: Any, num_decimal_digits: int = 1) -> None:
    """Format legend labels as scientific notation.

    Converts numeric legend labels to scientific notation format
    (e.g., 1000000 -> 1.0e+06).

    Args:
        g: Seaborn FacetGrid or similar object with a legend.
        num_decimal_digits: Number of decimal places in the mantissa.
    """
    # Round legend labels (hue values) to 3 significant figures.
    leg = getattr(g, "_legend", g.legend)
    if not hasattr(leg, "texts"):
        leg = g.legend_
    for txt in leg.texts:  # only the item labels, not the title
        try:
            txt.set_text(f"{float(txt.get_text()):.{num_decimal_digits}e}")
        except ValueError:
            pass  # skip any non-numeric labels


def format_g_legend_to_millions_and_billions(g: Any) -> None:
    """Format legend labels with M (millions) or B (billions) suffix.

    Converts numeric legend labels like "34061856.0" to "34M" or
    "1500000000" to "1B" for improved readability.

    Args:
        g: Seaborn FacetGrid, Axes, or similar object with a legend.
    """
    # FacetGrid keeps its legend on `_legend` and has no `get_legend()`; Axes has
    # `get_legend()` and no `_legend`. Accept either, matching this module's
    # `format_g_legend_in_scientific_notation`.
    legend = getattr(g, "_legend", None)
    if legend is None and hasattr(g, "get_legend"):
        legend = g.get_legend()
    if legend is None or not hasattr(legend, "get_texts"):
        return

    for text_obj in legend.get_texts():
        old_label_str = text_obj.get_text()

        try:
            num = float(old_label_str)
        except ValueError:
            continue  # not a numeric label (e.g. the legend title)

        # Leave sub-million values alone rather than falling through with a stale label.
        if num >= 1e9:
            text_obj.set_text(f"{int(num / 1e9)}B")
        elif num >= 1e6:
            # Truncate, so 62.8M renders as "62M".
            text_obj.set_text(f"{int(num / 1e6)}M")


def save_plot_with_multiple_extensions(
    plot_dir: str, plot_filename: str, use_tight_layout: bool = True
) -> None:
    """Save the current figure as both PDF and PNG.

    Saves the current matplotlib figure to the specified directory with
    both .pdf and .png extensions. PDF is preferred for publications
    while PNG is useful for quick viewing.

    Args:
        plot_dir: Directory to save the figures.
        plot_filename: Base filename without extension.
        use_tight_layout: If True, apply tight_layout before saving.
    """
    if use_tight_layout:
        # Ensure that axis labels don't overlap.
        plt.gcf().tight_layout()

    extensions = [
        "pdf",
        "png",
    ]
    for extension in extensions:
        plot_path = os.path.join(plot_dir, plot_filename + f".{extension}")
        print(f"Plotted {plot_path}")
        plt.savefig(plot_path, bbox_inches="tight", dpi=300)
