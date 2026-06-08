from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = ROOT / "data/analysis/qwen_harm_only_d1_d2_scores"
SUMMARY_JSON = ANALYSIS_DIR / "summary.json"
SCORES_CSV = ANALYSIS_DIR / "harm_only_max_scores.csv"
STRICT_SAFE_REFUSAL = (
    ROOT
    / "data/responses/d2_ood_2x2_safe_refusal_quick_scan/qwen/safe_refusal_quick_scan_confirmed_candidates.jsonl"
)
SHARED_OOD_SCORES = ROOT / "data/analysis/qwen_d1_d2_comparison/shared_ood_scores.csv"
JBB_ACTIVATIONS = ROOT / "data/activations/ood_clean/qwen/ood_jailbreakbench_clean.pt"
D1_PROBE = ROOT / "data/probes/itw_manual_d1/qwen.pt"
D2_PROBE = ROOT / "data/probes/itw_manual_d2/qwen.pt"
OUT_DIR = ROOT / "data/figures/presentation/d1_vs_d2"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, -80, 80)
    return 1.0 / (1.0 + np.exp(-values))


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


def get_variant_summary() -> dict[str, object]:
    summary = json.loads(SUMMARY_JSON.read_text(encoding="utf-8"))
    return summary["variants"]["harm_only_max"]


def probe_proba(layer_payload: dict, acts: torch.Tensor, layer: int) -> np.ndarray:
    x = acts[:, layer, :].to(torch.float32).numpy()
    mean = layer_payload["scaler_mean"].to(torch.float32).numpy()
    scale = layer_payload["scaler_scale"].to(torch.float32).numpy()
    coef = layer_payload["coef"].to(torch.float32).numpy()
    intercept = float(layer_payload["intercept"])
    return sigmoid(((x - mean) / scale) @ coef + intercept)


def strict_safe_refusal_scores() -> list[dict[str, object]]:
    rows = read_jsonl(STRICT_SAFE_REFUSAL)
    if not rows:
        return []
    ids = [row["id"] for row in rows]
    act_data = torch.load(JBB_ACTIVATIONS, weights_only=False)
    index = {rec_id: idx for idx, rec_id in enumerate(act_data["ids"])}
    missing = [rec_id for rec_id in ids if rec_id not in index]
    if missing:
        raise ValueError(f"missing strict safe-refusal activations: {missing}")

    acts = torch.stack([act_data["activations"][index[rec_id]] for rec_id in ids]).to(torch.float32)
    d1 = torch.load(D1_PROBE, weights_only=False)
    d2 = torch.load(D2_PROBE, weights_only=False)
    d1_scores = probe_proba(d1["layers"][25], acts, 25)
    d2_scores = probe_proba(d2["layers"][18], acts, 18)
    out = []
    for idx, row in enumerate(rows):
        out.append(
            {
                "id": row["id"],
                "source": row.get("source", ""),
                "target_cell": "safe|refusal",
                "d1_layer25_p_harm": float(d1_scores[idx]),
                "d2_layer18_p_refusal": float(d2_scores[idx]),
            }
        )
    return out


def old_safe_refusal_scores() -> list[dict[str, object]]:
    rows = read_csv(SHARED_OOD_SCORES)
    out = []
    for row in rows:
        if row["prompt_label"] == "safe" and row["behavior_label"] == "refusal":
            out.append(
                {
                    "id": row["id"],
                    "source": row.get("source", ""),
                    "target_cell": "safe|refusal",
                    "d1_layer25_p_harm": float(row["d1_p_harm"]),
                    "d2_layer18_p_refusal": float(row["d2_p_refusal"]),
                }
            )
    return out


def shared_ood_cell_scores(prompt_label: str, behavior_label: str) -> list[dict[str, object]]:
    rows = read_csv(SHARED_OOD_SCORES)
    out = []
    target_cell = f"{prompt_label}|{behavior_label}"
    for row in rows:
        if row["prompt_label"] == prompt_label and row["behavior_label"] == behavior_label:
            out.append(
                {
                    "id": row["id"],
                    "source": row.get("source", ""),
                    "target_cell": target_cell,
                    "d1_layer25_p_harm": float(row["d1_p_harm"]),
                    "d2_layer18_p_refusal": float(row["d2_p_refusal"]),
                }
            )
    return out


def make_behavior_predictor_bar(*, include_diagnostic: bool) -> None:
    setup_paper_style()
    variant = get_variant_summary()
    d1 = float(variant["d1_layer25_harm_score_as_behavior_predictor"]["auroc_behavior"])
    d2 = float(variant["d2_layer18_refusal_score_as_behavior_predictor"]["auroc_behavior"])
    d2_diag = float(variant["d2_layer27_refusal_score_as_behavior_predictor"]["auroc_behavior"])
    corr = variant["score_correlations"]["d1_layer25_vs_d2_layer18"]
    pearson = float(corr["pearson"])
    spearman = float(corr["spearman"])
    n = int(variant["n"])

    labels = ["D1 harm-score\n(layer 25)", "D2 refusal-score\n(layer 18)"]
    values = [d1, d2]
    colors = ["#f4a261", "#8ecae6"]
    if include_diagnostic:
        labels.append("D2 diagnostic\n(layer 27)")
        values.append(d2_diag)
        colors.append("#b7d7a8")

    fig, ax = plt.subplots(figsize=(5.4 if not include_diagnostic else 6.6, 3.2))
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=colors, edgecolor="#2f3e46", linewidth=0.8, width=0.58)
    ax.axhline(0.5, color="#777777", linestyle="--", linewidth=1.0, label="Chance")

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.018,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.text(
        0.02,
        0.94,
        f"harm-only OOD diagnostic, n={n}\n"
        f"D1 L25 vs D2 L18: Pearson {pearson:.3f}, Spearman {spearman:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.28", "fc": "white", "ec": "#cccccc", "alpha": 0.94},
    )

    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("AUROC for behavior label")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("D1 vs D2 on harm-only refusal/compliance")
    ax.grid(axis="y", linestyle="--", alpha=0.35, linewidth=0.7)
    handles = [
        Line2D([0], [0], color="#777777", linestyle="--", linewidth=1.0, label="Chance"),
        Patch(facecolor="#f4a261", edgecolor="#2f3e46", label="Prompt-risk score"),
        Patch(facecolor="#8ecae6", edgecolor="#2f3e46", label="Response-behavior score"),
    ]
    if include_diagnostic:
        handles.append(Patch(facecolor="#b7d7a8", edgecolor="#2f3e46", label="Diagnostic upper reference"))
    ax.legend(handles=handles, loc="lower right", frameon=True)
    fig.tight_layout()
    suffix = "with_diagnostic" if include_diagnostic else "primary"
    save(fig, f"d1_vs_d2_harm_only_behavior_auroc_{suffix}_paper")


def make_score_scatter() -> None:
    setup_paper_style()
    rows = read_csv(SCORES_CSV)
    d1 = np.array([float(r["d1_layer25_p_harm"]) for r in rows])
    d2 = np.array([float(r["d2_layer18_p_refusal"]) for r in rows])
    labels = np.array([r["behavior_label"] for r in rows])
    is_refusal = labels == "refusal"
    is_compliance = labels == "compliance"

    high_d1_low_d2 = (d1 >= 0.8) & (d2 <= 0.2) & is_compliance
    high_d1_high_d2 = (d1 >= 0.8) & (d2 >= 0.8) & is_refusal

    fig, ax = plt.subplots(figsize=(5.7, 4.15))
    ax.scatter(
        d1[is_compliance],
        d2[is_compliance],
        s=38,
        color="#D55E00",
        edgecolor="#2f3e46",
        linewidth=0.45,
        alpha=0.88,
        label="Observed compliance",
    )
    ax.scatter(
        d1[is_refusal],
        d2[is_refusal],
        s=38,
        color="#0072B2",
        edgecolor="#2f3e46",
        linewidth=0.45,
        alpha=0.88,
        label="Observed refusal",
    )

    ax.axvspan(0.8, 1.0, ymin=0.0, ymax=0.2, color="#D55E00", alpha=0.08, linewidth=0)
    ax.axhline(0.5, color="#888888", linestyle="--", linewidth=0.9)
    ax.axvline(0.5, color="#888888", linestyle="--", linewidth=0.9)
    ax.text(
        0.82,
        0.04,
        f"high D1,\nlow D2\ncompliance: {int(high_d1_low_d2.sum())}",
        fontsize=8,
        ha="left",
        va="bottom",
        color="#7f2704",
    )
    ax.text(
        0.04,
        0.96,
        f"high D1 & high D2 refusal: {int(high_d1_high_d2.sum())}",
        fontsize=8,
        ha="left",
        va="top",
        color="#023e8a",
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#dddddd", "alpha": 0.92},
    )

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("D1 harm-score (layer 25)")
    ax.set_ylabel("D2 refusal-score (layer 18)")
    ax.set_title("Recognition/action gap diagnostic")
    ax.grid(True, linestyle="--", alpha=0.28, linewidth=0.7)
    ax.legend(loc="lower left", frameon=True)
    fig.tight_layout()
    save(fig, "d1_vs_d2_harm_only_score_scatter_paper")


def make_cell_score_profile(*, safe_refusal_mode: str) -> None:
    setup_paper_style()
    rows = read_csv(SCORES_CSV)
    if safe_refusal_mode == "strict":
        safe_rows = strict_safe_refusal_scores()
        safe_label = "safe|refusal\nstrict"
        file_suffix = "strict_safe_refusal"
        note = "safe|refusal is exploratory: strict clean n=1."
    elif safe_refusal_mode == "old_ood":
        safe_rows = old_safe_refusal_scores()
        safe_label = "safe|refusal\nold OOD"
        file_suffix = "old_ood_safe_refusal"
        note = "safe|refusal is exploratory: old OOD pool, n=4."
    else:
        raise ValueError(safe_refusal_mode)

    cells = [
        ("harm|refusal", "harm|refusal", "#0072B2"),
        ("harm|compliance", "harm|compliance", "#D55E00"),
        (safe_label, "safe|refusal", "#009E73"),
    ]
    score_keys = [
        ("D1 harm-score", "d1_layer25_p_harm", "#f4a261"),
        ("D2 refusal-score", "d2_layer18_p_refusal", "#8ecae6"),
    ]

    grouped: dict[str, list[dict[str, object]]] = {
        "harm|refusal": [r for r in rows if r["target_cell"] == "harm|refusal"],
        "harm|compliance": [r for r in rows if r["target_cell"] == "harm|compliance"],
        "safe|refusal": safe_rows,
    }

    fig, ax = plt.subplots(figsize=(7.5, 3.65))
    x = np.arange(len(cells))
    offsets = [-0.18, 0.18]
    width = 0.30
    rng = np.random.default_rng(7)

    summary_rows = []
    for cell_idx, (label, cell, _) in enumerate(cells):
        cell_rows = grouped[cell]
        for score_idx, (score_label, key, color) in enumerate(score_keys):
            values = np.array([float(r[key]) for r in cell_rows], dtype=float)
            if len(values) == 0:
                continue
            median = float(np.median(values))
            mean = float(np.mean(values))
            xpos = x[cell_idx] + offsets[score_idx]
            ax.bar(
                xpos,
                median,
                width,
                color=color,
                edgecolor="#2f3e46",
                linewidth=0.8,
                alpha=0.72,
            )
            jitter = rng.normal(0, 0.026, size=len(values))
            ax.scatter(
                np.full(len(values), xpos) + jitter,
                values,
                s=24 if len(values) > 1 else 54,
                color=color,
                edgecolor="#2f3e46",
                linewidth=0.35,
                alpha=0.86,
                zorder=3,
            )
            ax.text(
                xpos,
                min(0.98, median + 0.045),
                f"{median:.2f}",
                ha="center",
                va="bottom",
                fontsize=7.5,
            )
            summary_rows.append(
                {
                    "cell": cell,
                    "score": score_label,
                    "n": len(values),
                    "mean": f"{mean:.6f}",
                    "median": f"{median:.6f}",
                    "min": f"{float(values.min()):.6f}",
                    "max": f"{float(values.max()):.6f}",
                }
            )
        ax.text(
            x[cell_idx],
            -0.115,
            f"n={len(cell_rows)}",
            ha="center",
            va="top",
            fontsize=8,
            transform=ax.get_xaxis_transform(),
        )

    ax.axhline(0.5, color="#777777", linestyle="--", linewidth=1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Probe score")
    ax.set_xticks(x)
    ax.set_xticklabels([label for label, _, _ in cells])
    ax.set_title("D1/D2 score profiles by prompt/behavior cell")
    ax.grid(axis="y", linestyle="--", alpha=0.35, linewidth=0.7)
    handles = [
        Line2D([0], [0], color="#777777", linestyle="--", linewidth=1.0, label="0.5 threshold"),
        Patch(facecolor="#f4a261", edgecolor="#2f3e46", alpha=0.72, label="D1 harm-score"),
        Patch(facecolor="#8ecae6", edgecolor="#2f3e46", alpha=0.72, label="D2 refusal-score"),
    ]
    ax.legend(handles=handles, loc="lower left", frameon=True)
    ax.text(
        0.985,
        0.04,
        "Bars show medians; dots show examples.\n" + note,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#cccccc", "alpha": 0.93},
    )
    fig.tight_layout()
    save(fig, f"d1_vs_d2_score_profiles_by_cell_{file_suffix}_paper")

    with (
        OUT_DIR / f"d1_vs_d2_score_profiles_by_cell_{file_suffix}_values.csv"
    ).open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["cell", "score", "n", "mean", "median", "min", "max"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)


def make_core_cell_score_profile(*, include_safe_refusal: bool = False) -> None:
    setup_paper_style()
    harm_rows = read_csv(SCORES_CSV)
    safe_compliance_rows = shared_ood_cell_scores("safe", "compliance")
    safe_refusal_rows = old_safe_refusal_scores() if include_safe_refusal else []

    cells = [
        ("harm|refusal", "harm|refusal"),
        ("harm|compliance", "harm|compliance"),
        ("safe|compliance", "safe|compliance"),
    ]
    if include_safe_refusal:
        cells.append(("safe|refusal\nexploratory", "safe|refusal"))

    grouped: dict[str, list[dict[str, object]]] = {
        "harm|refusal": [r for r in harm_rows if r["target_cell"] == "harm|refusal"],
        "harm|compliance": [r for r in harm_rows if r["target_cell"] == "harm|compliance"],
        "safe|compliance": safe_compliance_rows,
        "safe|refusal": safe_refusal_rows,
    }

    score_keys = [
        ("D1 harm-score", "d1_layer25_p_harm", "#f4a261"),
        ("D2 refusal-score", "d2_layer18_p_refusal", "#8ecae6"),
    ]
    fig_width = 7.7 if include_safe_refusal else 6.8
    fig, ax = plt.subplots(figsize=(fig_width, 3.65))
    x = np.arange(len(cells))
    offsets = [-0.18, 0.18]
    width = 0.30
    rng = np.random.default_rng(11)

    summary_rows = []
    for cell_idx, (label, cell) in enumerate(cells):
        cell_rows = grouped[cell]
        for score_idx, (score_label, key, color) in enumerate(score_keys):
            values = np.array([float(r[key]) for r in cell_rows], dtype=float)
            if len(values) == 0:
                continue
            median = float(np.median(values))
            mean = float(np.mean(values))
            xpos = x[cell_idx] + offsets[score_idx]
            ax.bar(
                xpos,
                median,
                width,
                color=color,
                edgecolor="#2f3e46",
                linewidth=0.8,
                alpha=0.72,
            )
            jitter = rng.normal(0, 0.026, size=len(values))
            ax.scatter(
                np.full(len(values), xpos) + jitter,
                values,
                s=22 if len(values) > 1 else 54,
                color=color,
                edgecolor="#2f3e46",
                linewidth=0.32,
                alpha=0.78,
                zorder=3,
            )
            ax.text(
                xpos,
                min(0.98, median + 0.045),
                f"{median:.2f}",
                ha="center",
                va="bottom",
                fontsize=7.5,
            )
            summary_rows.append(
                {
                    "cell": cell,
                    "score": score_label,
                    "n": len(values),
                    "mean": f"{mean:.6f}",
                    "median": f"{median:.6f}",
                    "min": f"{float(values.min()):.6f}",
                    "max": f"{float(values.max()):.6f}",
                }
            )
        ax.text(
            x[cell_idx],
            -0.12,
            f"n={len(cell_rows)}",
            ha="center",
            va="top",
            fontsize=8,
            transform=ax.get_xaxis_transform(),
        )

    ax.axhline(0.5, color="#777777", linestyle="--", linewidth=1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Probe score")
    ax.set_xticks(x)
    ax.set_xticklabels([label for label, _ in cells])
    ax.set_title("D1/D2 score profiles by prompt/behavior cell")
    ax.grid(axis="y", linestyle="--", alpha=0.35, linewidth=0.7)
    handles = [
        Line2D([0], [0], color="#777777", linestyle="--", linewidth=1.0, label="0.5 threshold"),
        Patch(facecolor="#f4a261", edgecolor="#2f3e46", alpha=0.72, label="D1 harm-score"),
        Patch(facecolor="#8ecae6", edgecolor="#2f3e46", alpha=0.72, label="D2 refusal-score"),
    ]
    ax.legend(handles=handles, loc="lower left", frameon=True)
    if include_safe_refusal:
        ax.text(
            0.985,
            0.04,
            "Bars show medians; dots show examples.\n"
            "safe|refusal is exploratory: old OOD pool, n=4.",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#cccccc", "alpha": 0.93},
        )
    fig.tight_layout()
    suffix = "core_plus_safe_refusal" if include_safe_refusal else "core"
    save(fig, f"d1_vs_d2_score_profiles_by_cell_{suffix}_paper")

    with (OUT_DIR / f"d1_vs_d2_score_profiles_by_cell_{suffix}_values.csv").open(
        "w", encoding="utf-8", newline=""
    ) as f:
        fieldnames = ["cell", "score", "n", "mean", "median", "min", "max"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)


def write_values_csv() -> None:
    variant = get_variant_summary()
    rows = [
        {
            "metric": "n",
            "value": variant["n"],
        },
        {
            "metric": "d1_layer25_behavior_auroc",
            "value": f"{float(variant['d1_layer25_harm_score_as_behavior_predictor']['auroc_behavior']):.6f}",
        },
        {
            "metric": "d2_layer18_behavior_auroc",
            "value": f"{float(variant['d2_layer18_refusal_score_as_behavior_predictor']['auroc_behavior']):.6f}",
        },
        {
            "metric": "d2_layer27_diagnostic_behavior_auroc",
            "value": f"{float(variant['d2_layer27_refusal_score_as_behavior_predictor']['auroc_behavior']):.6f}",
        },
        {
            "metric": "d1_l25_d2_l18_pearson",
            "value": f"{float(variant['score_correlations']['d1_layer25_vs_d2_layer18']['pearson']):.6f}",
        },
        {
            "metric": "d1_l25_d2_l18_spearman",
            "value": f"{float(variant['score_correlations']['d1_layer25_vs_d2_layer18']['spearman']):.6f}",
        },
    ]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / "d1_vs_d2_presentation_values.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    make_behavior_predictor_bar(include_diagnostic=False)
    make_behavior_predictor_bar(include_diagnostic=True)
    make_score_scatter()
    make_core_cell_score_profile(include_safe_refusal=False)
    make_core_cell_score_profile(include_safe_refusal=True)
    make_cell_score_profile(safe_refusal_mode="strict")
    make_cell_score_profile(safe_refusal_mode="old_ood")
    write_values_csv()
    print(OUT_DIR)
    for path in sorted(OUT_DIR.glob("*")):
        print(path)


if __name__ == "__main__":
    main()
