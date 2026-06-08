from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = ROOT / "data/validation/itw_manual_d1/qwen/text_baselines/metrics.json"
PERMUTATION_PATH = ROOT / "data/validation/itw_manual_d1/qwen/label_permutation/summary.json"
LAYER0_PATH = ROOT / "data/validation/itw_manual_d1/qwen/layer0_diagnostics/summary.json"
OUT_DIR = ROOT / "data/figures/presentation/dataset_validation/matplotlib"

COLORS = {
    "ink": "#4b4f56",
    "muted": "#7a7f87",
    "grid": "#d9dde3",
    "band": "#f2f4f7",
    "purple": "#c99aba",
    "blue": "#9cc7e8",
    "yellow": "#f5d67b",
    "coral": "#ee9999",
    "green": "#96c779",
}


def load_values() -> tuple[list[dict[str, object]], dict[str, object]]:
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    permutation = json.loads(PERMUTATION_PATH.read_text(encoding="utf-8"))
    layer_summary = json.loads(LAYER0_PATH.read_text(encoding="utf-8"))

    raw_token = metrics["raw_length_scores"]["qwen_token_count_as_harm_score"]["auroc"]
    token_best_direction = max(raw_token, 1.0 - raw_token)

    rows: list[dict[str, object]] = [
        {
            "label": "Перемешанные метки",
            "short": "shuffled labels",
            "value": permutation["overall_mean_auroc"],
            "color": COLORS["purple"],
        },
        {
            "label": "Только token count",
            "short": "token count",
            "value": token_best_direction,
            "color": COLORS["blue"],
        },
        {
            "label": "Length/token LogReg",
            "short": "length/token",
            "value": metrics["length_token_logreg"]["auroc"],
            "color": COLORS["yellow"],
        },
        {
            "label": "TF-IDF LogReg",
            "short": "TF-IDF",
            "value": metrics["tfidf_logreg"]["auroc"],
            "color": COLORS["coral"],
        },
    ]
    activation = {
        "label": "D1 activation probe",
        "short": "D1 activation",
        "value": layer_summary["layer12_metrics"]["auroc"],
        "color": COLORS["green"],
    }
    return rows, activation


def common_setup() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 24,
            "axes.labelsize": 14,
            "xtick.labelsize": 13,
            "ytick.labelsize": 15,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def save(fig: plt.Figure, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{name}.png", dpi=180, bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def make_barh(rows: list[dict[str, object]]) -> None:
    common_setup()
    fig, ax = plt.subplots(figsize=(13.33, 7.5))

    labels = [str(r["label"]) for r in rows]
    values = np.array([float(r["value"]) for r in rows])
    colors = [str(r["color"]) for r in rows]
    y = np.arange(len(rows))

    ax.axvspan(0.45, 0.55, color=COLORS["band"], zorder=0)
    ax.axvline(0.5, color=COLORS["ink"], linewidth=2.2)
    ax.barh(y, values - 0.5, left=0.5, color=colors, edgecolor="none", height=0.46)

    for yi, value in zip(y, values):
        ax.scatter(value, yi, s=90, color=COLORS["ink"], zorder=5)
        ax.text(value + 0.012, yi, f"{value:.3f}", va="center", ha="left", fontsize=16, weight="bold", color=COLORS["ink"])

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlim(0.45, 0.62)
    ax.set_xlabel("AUROC")
    ax.set_title("Проверка датасета: surface-baseline'ы около случайного уровня", loc="left", pad=18, color=COLORS["ink"], fontsize=22)
    fig.text(
        0.34,
        0.075,
        "ITW manual train/eval split: TF-IDF, длина и token count не дают уверенного разделения harm/safe.",
        fontsize=14,
        color=COLORS["muted"],
    )
    ax.grid(axis="x", color=COLORS["grid"], linewidth=1.0)
    ax.grid(axis="y", visible=False)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(COLORS["grid"])
    fig.subplots_adjust(left=0.32, right=0.96, top=0.82, bottom=0.20)

    save(fig, "matplotlib_surface_baselines_barh_ru")


def make_lollipop(rows: list[dict[str, object]]) -> None:
    common_setup()
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        pass

    fig, ax = plt.subplots(figsize=(13.33, 7.5))
    labels = [str(r["short"]) for r in rows]
    values = np.array([float(r["value"]) for r in rows])
    colors = [str(r["color"]) for r in rows]
    x = np.arange(len(rows))

    ax.axhspan(0.45, 0.55, color=COLORS["band"], zorder=0)
    ax.axhline(0.5, color=COLORS["ink"], linewidth=2.0)
    ax.vlines(x, 0.5, values, color=colors, linewidth=8, alpha=0.9)
    ax.scatter(x, values, s=220, color=colors, edgecolor=COLORS["ink"], linewidth=1.4, zorder=5)

    for xi, value in zip(x, values):
        ax.text(xi, value + 0.025, f"{value:.3f}", ha="center", va="bottom", fontsize=16, weight="bold", color=COLORS["ink"])

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontweight="bold")
    ax.set_ylim(0.42, 0.72)
    ax.set_ylabel("AUROC")
    ax.set_title("Surface-only checks do not separate the classes", loc="left", pad=18, color=COLORS["ink"], fontsize=22)
    ax.text(-0.38, 0.505, "chance", ha="left", va="bottom", fontsize=13, weight="bold", color=COLORS["ink"])
    ax.text(
        -0.38,
        0.435,
        "Все surface-baseline'ы остаются близко к 0.5; TF-IDF не приклеивается к потолку.",
        fontsize=14,
        color=COLORS["muted"],
    )
    ax.grid(axis="y", color=COLORS["grid"], linewidth=1.0)
    ax.grid(axis="x", visible=False)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.subplots_adjust(left=0.10, right=0.96, top=0.84, bottom=0.20)

    save(fig, "matplotlib_surface_baselines_lollipop_ru")


def make_surface_vs_activation(rows: list[dict[str, object]], activation: dict[str, object]) -> None:
    common_setup()
    fig, ax = plt.subplots(figsize=(13.33, 7.5))

    all_rows = rows + [activation]
    labels = [str(r["label"]) for r in all_rows]
    values = np.array([float(r["value"]) for r in all_rows])
    colors = [str(r["color"]) for r in all_rows]
    y = np.arange(len(all_rows))

    ax.axvspan(0.45, 0.55, color=COLORS["band"], zorder=0)
    ax.axvline(0.5, color=COLORS["ink"], linewidth=2.2)
    ax.barh(y, values - 0.5, left=0.5, color=colors, edgecolor="none", height=0.42)

    for yi, value in zip(y, values):
        ax.scatter(value, yi, s=90, color=COLORS["ink"], zorder=5)
        ax.text(value + 0.012, yi, f"{value:.3f}", va="center", ha="left", fontsize=15, weight="bold", color=COLORS["ink"])

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlim(0.45, 1.0)
    ax.set_xlabel("AUROC")
    ax.set_title("Surface-baseline'ы низкие, activation probe высокий", loc="left", pad=18, color=COLORS["ink"], fontsize=22)
    ax.text(
        0.45,
        len(all_rows) + 0.15,
        "Этот вариант лучше использовать на results-слайде, а не на слайде валидации датасета.",
        fontsize=14,
        color=COLORS["muted"],
    )
    ax.grid(axis="x", color=COLORS["grid"], linewidth=1.0)
    ax.grid(axis="y", visible=False)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(COLORS["grid"])
    fig.subplots_adjust(left=0.32, right=0.96, top=0.82, bottom=0.20)

    save(fig, "matplotlib_surface_vs_activation_barh_ru")


def make_paper_style_bar(rows: list[dict[str, object]], activation: dict[str, object]) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 9,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )

    labels = [str(r["short"]) for r in rows]
    values = np.array([float(r["value"]) for r in rows])
    x = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    ax.bar(
        x,
        values,
        color="#8ecae6",
        edgecolor="#2f3e46",
        linewidth=0.9,
        label="Surface baseline",
        width=0.55,
    )
    ax.axhline(0.5, color="#d95f02", linestyle="--", linewidth=1.2, label="Chance")

    for xi, value in zip(x, values):
        ax.text(xi, value + 0.025, f"{value:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("AUROC")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=28, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.35, linewidth=0.7)
    ax.legend(title="Condition", loc="upper right", frameon=True, fontsize=8, title_fontsize=8)
    ax.set_title("Surface baselines on ITW manual eval")
    fig.tight_layout()
    save(fig, "matplotlib_paper_style_surface_baselines")

    labels2 = labels + [str(activation["short"])]
    values2 = np.array(list(values) + [float(activation["value"])])
    colors2 = ["#f4a261"] * len(rows) + ["#8ecae6"]
    x2 = np.arange(len(labels2))

    fig, ax = plt.subplots(figsize=(7.6, 3.0))
    ax.bar(
        x2,
        values2,
        color=colors2,
        edgecolor="#2f3e46",
        linewidth=0.9,
        width=0.55,
    )
    ax.axhline(0.5, color="#777777", linestyle="--", linewidth=1.1)

    for xi, value in zip(x2, values2):
        ax.text(xi, value + 0.025, f"{value:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("AUROC")
    ax.set_xticks(x2)
    ax.set_xticklabels(labels2, rotation=28, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.35, linewidth=0.7)
    handles = [
        Line2D([0], [0], color="#777777", linestyle="--", linewidth=1.1, label="Chance"),
        Patch(facecolor="#f4a261", edgecolor="#2f3e46", label="Surface baseline"),
        Patch(facecolor="#8ecae6", edgecolor="#2f3e46", label="Activation probe"),
    ]
    ax.legend(handles=handles, title="Condition", loc="upper left", frameon=True, fontsize=8, title_fontsize=8)
    ax.set_title("Text baselines vs activation probe")
    fig.tight_layout()
    save(fig, "matplotlib_paper_style_surface_vs_activation")


def main() -> None:
    rows, activation = load_values()
    make_barh(rows)
    make_lollipop(rows)
    make_surface_vs_activation(rows, activation)
    make_paper_style_bar(rows, activation)
    print(OUT_DIR)
    for path in sorted(OUT_DIR.glob("*")):
        print(path)


if __name__ == "__main__":
    main()
