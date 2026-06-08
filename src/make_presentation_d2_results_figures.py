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
D2_DIR = ROOT / "data/validation/itw_manual_d2/qwen/harm_only_dev_test_layer_selection"
LAYER_CSV = D2_DIR / "qwen_d2_harm_only_dev_test_layers.csv"
SUMMARY_JSON = D2_DIR / "qwen_d2_harm_only_dev_test_summary.json"
OUT_DIR = ROOT / "data/figures/presentation/d2_results"


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


def load_variant(variant: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    rows = [r for r in read_csv(LAYER_CSV) if r["variant"] == variant]
    summary = json.loads(SUMMARY_JSON.read_text(encoding="utf-8"))
    layers = np.array([int(r["layer"]) for r in rows])
    dev = np.array([float(r["dev_auroc"]) for r in rows])
    test = np.array([float(r["test_auroc"]) for r in rows])
    return layers, dev, test, summary["variants"][variant]


def make_layerwise_plot(*, variant: str = "harm_only_max", full_scale: bool = False) -> None:
    setup_paper_style()
    layers, dev, test, summary = load_variant(variant)
    selected_layer = int(summary["selected_layer_by_dev_auroc"])
    selected_dev = float(summary["selected_layer_dev_metrics"]["auroc"])
    selected_test = float(summary["selected_layer_test_metrics"]["auroc"])
    dev_n = int(summary["dev"]["n"])
    test_n = int(summary["test"]["n"])

    fig, ax = plt.subplots(figsize=(7.6, 3.15))
    ax.plot(layers, dev, marker="o", markersize=3.4, linewidth=1.4, color="#0072B2", label="harm-only dev")
    ax.plot(layers, test, marker="s", markersize=3.2, linewidth=1.4, color="#D55E00", label="harm-only test")
    ax.axhline(0.5, color="#777777", linestyle="--", linewidth=1.0, label="Chance")
    ax.axvline(selected_layer, color="#333333", linestyle=":", linewidth=1.3)
    ax.scatter([selected_layer], [selected_dev], s=52, color="#0072B2", edgecolor="#222222", zorder=5)
    ax.scatter([selected_layer], [selected_test], s=52, color="#D55E00", edgecolor="#222222", zorder=5)

    ax.annotate(
        f"selected layer {selected_layer}\n"
        f"dev {selected_dev:.3f}, test {selected_test:.3f}\n"
        f"dev n={dev_n}, test n={test_n}",
        xy=(selected_layer, selected_test),
        xytext=(selected_layer - 12.7, selected_test - 0.150),
        arrowprops={"arrowstyle": "->", "linewidth": 0.9, "color": "#333333"},
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#cccccc", "alpha": 0.92},
    )

    ax.set_xlim(-0.5, 27.5)
    ax.set_ylim(0.0, 1.0) if full_scale else ax.set_ylim(0.18, 1.0)
    ax.set_xlabel("Layer")
    ax.set_ylabel("AUROC")
    ax.set_title("D2 response-behavior probe: harm-only layer selection")
    ax.grid(axis="y", linestyle="--", alpha=0.35, linewidth=0.7)
    ax.grid(axis="x", alpha=0.10, linewidth=0.5)
    ax.legend(title="Split", loc="lower right", frameon=True, title_fontsize=8)
    fig.tight_layout()

    suffix = "fullscale" if full_scale else "zoom"
    save(fig, f"d2_{variant}_layerwise_dev_test_{suffix}_paper")


def make_variant_summary_bar() -> None:
    setup_paper_style()
    summary = json.loads(SUMMARY_JSON.read_text(encoding="utf-8"))["variants"]
    variants = ["source_matched", "harm_only_max"]
    labels = ["Source-matched\n(smaller)", "Harm-only max\n(primary)"]
    dev = [float(summary[v]["selected_layer_dev_metrics"]["auroc"]) for v in variants]
    test = [float(summary[v]["selected_layer_test_metrics"]["auroc"]) for v in variants]
    selected_layers = [int(summary[v]["selected_layer_by_dev_auroc"]) for v in variants]

    x = np.arange(len(variants))
    width = 0.34

    fig, ax = plt.subplots(figsize=(5.9, 3.15))
    bars1 = ax.bar(
        x - width / 2,
        dev,
        width,
        color="#8ecae6",
        edgecolor="#2f3e46",
        linewidth=0.8,
        label="dev AUROC",
    )
    bars2 = ax.bar(
        x + width / 2,
        test,
        width,
        color="#f4a261",
        edgecolor="#2f3e46",
        linewidth=0.8,
        label="test AUROC",
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
    for i, layer in enumerate(selected_layers):
        ax.text(i, 0.05, f"selected L{layer}", ha="center", va="bottom", fontsize=8)

    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("AUROC")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("D2 harm-only protocols")
    ax.grid(axis="y", linestyle="--", alpha=0.35, linewidth=0.7)
    handles = [
        Line2D([0], [0], color="#777777", linestyle="--", linewidth=1.0, label="Chance"),
        Patch(facecolor="#8ecae6", edgecolor="#2f3e46", label="dev AUROC"),
        Patch(facecolor="#f4a261", edgecolor="#2f3e46", label="test AUROC"),
    ]
    ax.legend(handles=handles, title="Split", loc="lower right", frameon=True, title_fontsize=8)
    fig.tight_layout()
    save(fig, "d2_harm_only_protocol_summary_paper")


def write_summary_csv() -> None:
    summary = json.loads(SUMMARY_JSON.read_text(encoding="utf-8"))["variants"]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / "d2_results_presentation_values.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["variant", "metric", "value"])
        writer.writeheader()
        for variant, data in summary.items():
            writer.writerow({"variant": variant, "metric": "selected_layer", "value": data["selected_layer_by_dev_auroc"]})
            writer.writerow({"variant": variant, "metric": "dev_n", "value": data["dev"]["n"]})
            writer.writerow({"variant": variant, "metric": "test_n", "value": data["test"]["n"]})
            writer.writerow({"variant": variant, "metric": "selected_dev_auroc", "value": f"{float(data['selected_layer_dev_metrics']['auroc']):.6f}"})
            writer.writerow({"variant": variant, "metric": "selected_test_auroc", "value": f"{float(data['selected_layer_test_metrics']['auroc']):.6f}"})
            writer.writerow({"variant": variant, "metric": "selected_test_accuracy", "value": f"{float(data['selected_layer_test_metrics']['accuracy']):.6f}"})
            writer.writerow({"variant": variant, "metric": "selected_test_f1", "value": f"{float(data['selected_layer_test_metrics']['f1']):.6f}"})


def main() -> None:
    make_layerwise_plot(variant="harm_only_max", full_scale=False)
    make_layerwise_plot(variant="harm_only_max", full_scale=True)
    make_layerwise_plot(variant="source_matched", full_scale=False)
    make_variant_summary_bar()
    write_summary_csv()
    print(OUT_DIR)
    for path in sorted(OUT_DIR.glob("*")):
        print(path)


if __name__ == "__main__":
    main()
