from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[1]
D1_DIR = ROOT / "data/validation/itw_manual_d1/qwen/ood_clean/ood_dev_test_layer_selection"
LAYER_SUMMARY = D1_DIR / "layer_selection_summary.csv"
SELECTED_TEST = D1_DIR / "selected_layer_test_metrics.csv"
SUMMARY_JSON = D1_DIR / "summary.json"
OUT_DIR = ROOT / "data/figures/presentation/d1_results"


TEXT_BASELINE_TEST_AUROC = {
    "ood_advbench_alpaca": 0.6655,
    "ood_harmbench_alpaca": 0.4779,
    "ood_jailbreakbench": 0.6172,
    "ood_xstest": 0.5805,
}

DATASET_LABELS = {
    "ood_advbench_alpaca": "AdvBench\n+ Alpaca",
    "ood_harmbench_alpaca": "HarmBench\n+ Alpaca",
    "ood_jailbreakbench": "JailbreakBench",
    "ood_xstest": "XSTest",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def setup_paper_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def save(fig: plt.Figure, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{name}.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def load_layer_summary() -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    rows = read_csv(LAYER_SUMMARY)
    summary = json.loads(SUMMARY_JSON.read_text(encoding="utf-8"))
    layers = np.array([int(r["layer"]) for r in rows])
    dev = np.array([float(r["mean_dev_auroc"]) for r in rows])
    test = np.array([float(r["mean_test_auroc"]) for r in rows])
    return layers, dev, test, summary


def make_layerwise_plot(*, full_scale: bool = False) -> None:
    setup_paper_style()
    layers, dev, test, summary = load_layer_summary()
    selected_layer = int(summary["selected_layer"])
    selected_dev = float(summary["selected_layer_mean_dev_auroc"])
    selected_test = float(summary["selected_layer_mean_test_auroc"])

    fig, ax = plt.subplots(figsize=(7.6, 3.15))
    ax.plot(layers, dev, marker="o", markersize=3.4, linewidth=1.4, color="#0072B2", label="OOD-dev mean")
    ax.plot(layers, test, marker="s", markersize=3.2, linewidth=1.4, color="#D55E00", label="OOD-test mean")
    ax.axhline(0.5, color="#777777", linestyle="--", linewidth=1.0, label="Chance")
    ax.axvline(selected_layer, color="#333333", linestyle=":", linewidth=1.3)
    ax.scatter([selected_layer], [selected_dev], s=52, color="#0072B2", edgecolor="#222222", zorder=5)
    ax.scatter([selected_layer], [selected_test], s=52, color="#D55E00", edgecolor="#222222", zorder=5)

    ax.annotate(
        f"selected layer {selected_layer}\n"
        f"dev {selected_dev:.3f}, test {selected_test:.3f}",
        xy=(selected_layer, selected_test),
        xytext=(selected_layer - 9.8, selected_test - 0.025),
        arrowprops={"arrowstyle": "->", "linewidth": 0.9, "color": "#333333"},
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#cccccc", "alpha": 0.92},
    )

    ax.set_xlim(-0.5, 27.5)
    ax.set_ylim(0.0, 1.0) if full_scale else ax.set_ylim(0.42, 0.90)
    ax.set_xlabel("Layer")
    ax.set_ylabel("AUROC")
    ax.set_title("D1 prompt-risk probe: OOD-dev layer selection")
    ax.grid(axis="y", linestyle="--", alpha=0.35, linewidth=0.7)
    ax.grid(axis="x", alpha=0.10, linewidth=0.5)
    ax.legend(title="Split", loc="lower right", frameon=True, title_fontsize=8)
    fig.tight_layout()
    suffix = "fullscale" if full_scale else "zoom"
    save(fig, f"d1_layerwise_ood_dev_test_{suffix}_paper")


def make_selected_layer_bar_plot() -> None:
    setup_paper_style()
    rows = read_csv(SELECTED_TEST)
    datasets = [r["dataset"] for r in rows]
    probe = np.array([float(r["auroc"]) for r in rows])
    tfidf = np.array([TEXT_BASELINE_TEST_AUROC[d] for d in datasets])

    x = np.arange(len(datasets))
    width = 0.34

    fig, ax = plt.subplots(figsize=(7.6, 3.15))
    bars1 = ax.bar(
        x - width / 2,
        tfidf,
        width,
        color="#f4a261",
        edgecolor="#2f3e46",
        linewidth=0.8,
        label="TF-IDF baseline",
    )
    bars2 = ax.bar(
        x + width / 2,
        probe,
        width,
        color="#8ecae6",
        edgecolor="#2f3e46",
        linewidth=0.8,
        label="D1 activation probe",
    )
    ax.axhline(0.5, color="#777777", linestyle="--", linewidth=1.0, label="Chance")

    for bars in [bars1, bars2]:
        for bar in bars:
            value = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.018,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=7.5,
            )

    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("AUROC")
    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABELS[d] for d in datasets], rotation=22, ha="right")
    ax.set_title("D1 selected layer 25: OOD-test transfer")
    ax.grid(axis="y", linestyle="--", alpha=0.35, linewidth=0.7)
    handles = [
        Line2D([0], [0], color="#777777", linestyle="--", linewidth=1.0, label="Chance"),
        Patch(facecolor="#f4a261", edgecolor="#2f3e46", label="TF-IDF baseline"),
        Patch(facecolor="#8ecae6", edgecolor="#2f3e46", label="D1 activation probe"),
    ]
    ax.legend(handles=handles, title="Condition", loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=True, title_fontsize=8)
    fig.tight_layout()
    save(fig, "d1_selected_layer25_ood_test_vs_tfidf_paper")


def write_summary_csv() -> None:
    _, _, _, summary = load_layer_summary()
    rows = read_csv(SELECTED_TEST)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / "d1_results_presentation_values.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "dataset", "value"])
        writer.writeheader()
        writer.writerow({"metric": "selected_layer", "dataset": "mean_ood_dev", "value": summary["selected_layer"]})
        writer.writerow({"metric": "mean_dev_auroc", "dataset": "mean_ood_dev", "value": f"{float(summary['selected_layer_mean_dev_auroc']):.6f}"})
        writer.writerow({"metric": "mean_test_auroc", "dataset": "mean_ood_test", "value": f"{float(summary['selected_layer_mean_test_auroc']):.6f}"})
        for row in rows:
            dataset = row["dataset"]
            writer.writerow({"metric": "d1_layer25_test_auroc", "dataset": dataset, "value": f"{float(row['auroc']):.6f}"})
            writer.writerow({"metric": "tfidf_test_auroc", "dataset": dataset, "value": f"{TEXT_BASELINE_TEST_AUROC[dataset]:.6f}"})


def main() -> None:
    make_layerwise_plot(full_scale=False)
    make_layerwise_plot(full_scale=True)
    make_selected_layer_bar_plot()
    write_summary_csv()
    print(OUT_DIR)
    for path in sorted(OUT_DIR.glob("*")):
        print(path)


if __name__ == "__main__":
    main()
