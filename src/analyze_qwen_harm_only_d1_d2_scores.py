"""
Analyze Qwen D1 harm-score vs D2 refusal-score on harm-only interim rows.

This is a score diagnostic, not a new training stage:

- D1 remains a prompt-risk probe (`harm` vs `safe`).
- D2 remains a response-behavior probe (`refusal` vs `compliance`).

The key question is whether, inside prompts already labeled `harm`, the D2
refusal-score tracks observed refusal/compliance better than the D1 harm-score.
That would support a recognition/action-gap interpretation:

    high D1 harm-score + low D2 refusal-score + observed compliance
"""

import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path


ROOT = Path("C:/Users/Vladi/Desktop/projs/interpetability")
DEPS = ROOT / ".deps" / "python"
if DEPS.exists():
    sys.path.insert(0, str(DEPS))

import numpy as np
import torch
from sklearn.metrics import roc_auc_score


D1_PROBE = ROOT / "data/probes/itw_manual_d1/qwen.pt"
D2_PROBE = ROOT / "data/probes/itw_manual_d2/qwen.pt"
LABEL_FILES = {
    "source_matched": ROOT
    / "data/responses/d2_ood_harm_stratified_interim/qwen/qwen_d2_ood_harm_only_source_matched_labeled.jsonl",
    "harm_only_max": ROOT
    / "data/responses/d2_ood_harm_stratified_interim/qwen/qwen_d2_ood_harm_only_max_labeled.jsonl",
}
ACTIVATION_FILES = [
    ROOT / "data/activations/ood_clean/qwen/ood_advbench_alpaca_clean.pt",
    ROOT / "data/activations/ood_clean/qwen/ood_harmbench_alpaca_clean.pt",
    ROOT / "data/activations/ood_clean/qwen/ood_jailbreakbench_clean.pt",
    ROOT / "data/activations/ood_clean/qwen/ood_xstest_clean.pt",
]
OUT_DIR = ROOT / "data/analysis/qwen_harm_only_d1_d2_scores"


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, -80, 80)
    return 1.0 / (1.0 + np.exp(-values))


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and sorted_values[j] == sorted_values[i]:
            j += 1
        avg_rank = (i + j - 1) / 2.0
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def corr_pair(x: np.ndarray, y: np.ndarray) -> dict:
    if len(x) < 2 or float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return {"pearson": math.nan, "spearman": math.nan}
    return {
        "pearson": float(np.corrcoef(x, y)[0, 1]),
        "spearman": float(np.corrcoef(rankdata(x), rankdata(y))[0, 1]),
    }


def load_activation_index(paths: list[Path]) -> dict[str, torch.Tensor]:
    index = {}
    for path in paths:
        data = torch.load(path, weights_only=False)
        for idx, rec_id in enumerate(data["ids"]):
            if rec_id in index:
                raise ValueError(f"duplicate activation id: {rec_id}")
            index[rec_id] = data["activations"][idx]
    return index


def stack_activations(rows: list[dict], activation_index: dict[str, torch.Tensor]) -> torch.Tensor:
    missing = [row["id"] for row in rows if row["id"] not in activation_index]
    if missing:
        raise ValueError(f"missing activations for ids: {missing[:20]}")
    return torch.stack([activation_index[row["id"]] for row in rows]).to(torch.float32)


def probe_proba(layer_payload: dict, acts: torch.Tensor, layer: int) -> np.ndarray:
    x = acts[:, layer, :].to(torch.float32).numpy()
    mean = layer_payload["scaler_mean"].to(torch.float32).numpy()
    scale = layer_payload["scaler_scale"].to(torch.float32).numpy()
    coef = layer_payload["coef"].to(torch.float32).numpy()
    intercept = float(layer_payload["intercept"])
    return sigmoid(((x - mean) / scale) @ coef + intercept)


def clean_preview(text: str, n: int = 220) -> str:
    text = " ".join((text or "").split())
    if len(text) > n:
        text = text[:n] + "..."
    return text.encode("ascii", "backslashreplace").decode("ascii")


def score_summary(scores: np.ndarray, y_refusal: np.ndarray) -> dict:
    refusal_scores = scores[y_refusal == 1]
    compliance_scores = scores[y_refusal == 0]
    return {
        "mean_refusal": float(refusal_scores.mean()),
        "mean_compliance": float(compliance_scores.mean()),
        "median_refusal": float(np.median(refusal_scores)),
        "median_compliance": float(np.median(compliance_scores)),
        "mean_gap_refusal_minus_compliance": float(refusal_scores.mean() - compliance_scores.mean()),
        "auroc_behavior": float(roc_auc_score(y_refusal, scores)),
    }


def add_rank_columns(rows: list[dict], columns: list[str]) -> None:
    for column in columns:
        values = np.array([float(row[column]) for row in rows])
        # Highest score receives rank 1.
        descending_ranks = rankdata(-values) + 1.0
        for row, rank in zip(rows, descending_ranks):
            row[f"{column}_desc_rank"] = float(rank)


def top_cases(rows: list[dict], variant: str, out_dir: Path) -> dict:
    compliance = [row for row in rows if row["behavior_label"] == "compliance"]
    refusal = [row for row in rows if row["behavior_label"] == "refusal"]

    high_d1_low_d2_compliance = sorted(
        compliance,
        key=lambda row: (
            -float(row["d1_layer25_p_harm"]),
            float(row["d2_layer18_p_refusal"]),
            float(row["d2_layer27_p_refusal"]),
        ),
    )[:10]
    high_d1_high_d2_refusal = sorted(
        refusal,
        key=lambda row: (
            -float(row["d1_layer25_p_harm"]),
            -float(row["d2_layer18_p_refusal"]),
            -float(row["d2_layer27_p_refusal"]),
        ),
    )[:10]
    high_d1_high_d2_compliance = sorted(
        compliance,
        key=lambda row: (
            -float(row["d1_layer25_p_harm"]),
            -float(row["d2_layer18_p_refusal"]),
            -float(row["d2_layer27_p_refusal"]),
        ),
    )[:10]
    low_d1_high_d2_refusal = sorted(
        refusal,
        key=lambda row: (
            float(row["d1_layer25_p_harm"]),
            -float(row["d2_layer18_p_refusal"]),
            -float(row["d2_layer27_p_refusal"]),
        ),
    )[:10]

    case_sets = {
        "high_d1_low_d2_compliance": high_d1_low_d2_compliance,
        "high_d1_high_d2_refusal": high_d1_high_d2_refusal,
        "high_d1_high_d2_compliance": high_d1_high_d2_compliance,
        "low_d1_high_d2_refusal": low_d1_high_d2_refusal,
    }
    outputs = {}
    fields = [
        "variant",
        "id",
        "source",
        "behavior_label",
        "target_cell",
        "d1_layer25_p_harm",
        "d1_layer12_p_harm",
        "d2_layer18_p_refusal",
        "d2_layer20_p_refusal",
        "d2_layer27_p_refusal",
        "d1_layer25_p_harm_desc_rank",
        "d2_layer18_p_refusal_desc_rank",
        "d2_layer27_p_refusal_desc_rank",
        "prompt_preview",
    ]
    for name, case_rows in case_sets.items():
        path = out_dir / f"{variant}_{name}.csv"
        write_csv(path, case_rows, fields)
        outputs[name] = str(path.relative_to(ROOT))
    return outputs


def main() -> None:
    d1 = torch.load(D1_PROBE, weights_only=False)
    d2 = torch.load(D2_PROBE, weights_only=False)
    activation_index = load_activation_index(ACTIVATION_FILES)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_score_rows = []
    summary = {
        "d1_probe": str(D1_PROBE.relative_to(ROOT)),
        "d2_probe": str(D2_PROBE.relative_to(ROOT)),
        "activation_files": [str(path.relative_to(ROOT)) for path in ACTIVATION_FILES],
        "d1_layers": {
            "primary_ood_dev_selected": 25,
            "itw_selected_control": 12,
        },
        "d2_layers": {
            "ood_balanced_selected": 18,
            "itw_selected_control": 20,
            "harm_only_best_diagnostic": 27,
        },
        "variants": {},
        "note": (
            "D1 is not evaluated as a behavior probe here. Its harm-score is used "
            "only as a diagnostic predictor of behavior inside prompt_label=harm."
        ),
    }

    for variant, label_path in LABEL_FILES.items():
        rows = [
            row
            for row in read_jsonl(label_path)
            if row["prompt_label"] == "harm"
            and row["behavior_label"] in ("refusal", "compliance")
        ]
        rows.sort(key=lambda row: row["id"])
        acts = stack_activations(rows, activation_index)
        y_refusal = np.array([1 if row["behavior_label"] == "refusal" else 0 for row in rows])

        d1_l25 = probe_proba(d1["layers"][25], acts, 25)
        d1_l12 = probe_proba(d1["layers"][12], acts, 12)
        d2_l18 = probe_proba(d2["layers"][18], acts, 18)
        d2_l20 = probe_proba(d2["layers"][20], acts, 20)
        d2_l27 = probe_proba(d2["layers"][27], acts, 27)

        score_rows = []
        for idx, row in enumerate(rows):
            score_row = {
                "variant": variant,
                "id": row["id"],
                "source": row.get("source"),
                "prompt_label": row["prompt_label"],
                "behavior_label": row["behavior_label"],
                "target_cell": row.get("target_cell"),
                "y_refusal": int(y_refusal[idx]),
                "d1_layer25_p_harm": float(d1_l25[idx]),
                "d1_layer12_p_harm": float(d1_l12[idx]),
                "d2_layer18_p_refusal": float(d2_l18[idx]),
                "d2_layer20_p_refusal": float(d2_l20[idx]),
                "d2_layer27_p_refusal": float(d2_l27[idx]),
                "prompt_preview": clean_preview(row.get("prompt") or ""),
            }
            score_rows.append(score_row)
        add_rank_columns(
            score_rows,
            ["d1_layer25_p_harm", "d1_layer12_p_harm", "d2_layer18_p_refusal", "d2_layer20_p_refusal", "d2_layer27_p_refusal"],
        )
        all_score_rows.extend(score_rows)

        variant_csv = OUT_DIR / f"{variant}_scores.csv"
        write_csv(variant_csv, score_rows, list(score_rows[0].keys()))
        case_outputs = top_cases(score_rows, variant, OUT_DIR)

        summary["variants"][variant] = {
            "label_file": str(label_path.relative_to(ROOT)),
            "n": len(rows),
            "counts": {
                "behavior": dict(Counter(row["behavior_label"] for row in rows)),
                "source": dict(Counter(row.get("source") for row in rows)),
                "target_cell": dict(Counter(row.get("target_cell") for row in rows)),
            },
            "d1_layer25_harm_score_as_behavior_predictor": score_summary(d1_l25, y_refusal),
            "d1_layer12_harm_score_as_behavior_predictor": score_summary(d1_l12, y_refusal),
            "d2_layer18_refusal_score_as_behavior_predictor": score_summary(d2_l18, y_refusal),
            "d2_layer20_refusal_score_as_behavior_predictor": score_summary(d2_l20, y_refusal),
            "d2_layer27_refusal_score_as_behavior_predictor": score_summary(d2_l27, y_refusal),
            "score_correlations": {
                "d1_layer25_vs_d2_layer18": corr_pair(d1_l25, d2_l18),
                "d1_layer25_vs_d2_layer27": corr_pair(d1_l25, d2_l27),
                "d1_layer12_vs_d2_layer20": corr_pair(d1_l12, d2_l20),
            },
            "outputs": {
                "scores": str(variant_csv.relative_to(ROOT)),
                **case_outputs,
            },
        }

    all_csv = OUT_DIR / "all_harm_only_d1_d2_scores.csv"
    write_csv(all_csv, all_score_rows, list(all_score_rows[0].keys()))
    summary["outputs"] = {
        "all_scores": str(all_csv.relative_to(ROOT)),
        "summary": str((OUT_DIR / "summary.json").relative_to(ROOT)),
        "report": str((OUT_DIR / "report.md").relative_to(ROOT)),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report = [
        "# Qwen Harm-Only D1 vs D2 Score Diagnostic",
        "",
        "This diagnostic asks whether Qwen can show a high prompt-risk signal while still showing a low refusal-behavior signal.",
        "",
        "- D1 score: `P(harm)` from the prompt-risk probe.",
        "- D2 score: `P(refusal)` from the response-behavior probe.",
        "- Rows: only `prompt_label=harm`.",
        "",
    ]
    for variant, item in summary["variants"].items():
        d1_25 = item["d1_layer25_harm_score_as_behavior_predictor"]
        d2_18 = item["d2_layer18_refusal_score_as_behavior_predictor"]
        d2_27 = item["d2_layer27_refusal_score_as_behavior_predictor"]
        corr = item["score_correlations"]["d1_layer25_vs_d2_layer18"]
        report.extend(
            [
                f"## {variant}",
                "",
                f"- Counts: `{item['counts']['target_cell']}`",
                f"- D1 layer 25 harm-score as behavior predictor: AUROC `{d1_25['auroc_behavior']:.4f}`, mean refusal `{d1_25['mean_refusal']:.4f}`, mean compliance `{d1_25['mean_compliance']:.4f}`.",
                f"- D2 layer 18 refusal-score as behavior predictor: AUROC `{d2_18['auroc_behavior']:.4f}`, mean refusal `{d2_18['mean_refusal']:.4f}`, mean compliance `{d2_18['mean_compliance']:.4f}`.",
                f"- D2 layer 27 diagnostic refusal-score: AUROC `{d2_27['auroc_behavior']:.4f}`, mean refusal `{d2_27['mean_refusal']:.4f}`, mean compliance `{d2_27['mean_compliance']:.4f}`.",
                f"- D1 layer 25 vs D2 layer 18 score correlation: Pearson `{corr['pearson']:.4f}`, Spearman `{corr['spearman']:.4f}`.",
                "",
            ]
        )
    report.extend(
        [
            "## Interpretation",
            "",
            "If D1 harm-score is high for both refused and complied harmful prompts, while D2 refusal-score is much higher for refused prompts, that supports a recognition/action-gap reading.",
            "In that reading, Qwen can represent prompt risk without necessarily activating the refusal-behavior direction strongly enough to refuse.",
            "",
        ]
    )
    (OUT_DIR / "report.md").write_text("\n".join(report), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"wrote {(OUT_DIR / 'report.md').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
