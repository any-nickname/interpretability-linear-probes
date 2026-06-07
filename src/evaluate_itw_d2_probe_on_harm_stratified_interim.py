"""
Evaluate the ITW-trained Qwen D2 probe on the interim harm-stratified OOD set.

Primary diagnostic:
    harm|refusal vs harm|compliance

This keeps the prompt-risk label fixed to `harm`, so a good result cannot be
explained simply by harm-vs-safe prompt labels. No layer is selected here for a
new primary protocol; best-layer values are diagnostic.
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
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


PROBE = ROOT / "data/probes/itw_manual_d2/qwen.pt"
LABEL_FILES = {
    "source_matched": ROOT
    / "data/responses/d2_ood_harm_stratified_interim/qwen/qwen_d2_ood_harm_only_source_matched_labeled.jsonl",
    "harm_only_max": ROOT
    / "data/responses/d2_ood_harm_stratified_interim/qwen/qwen_d2_ood_harm_only_max_labeled.jsonl",
    "harm_only_max_plus_safe_compliance_control": ROOT
    / "data/responses/d2_ood_harm_stratified_interim/qwen/qwen_d2_ood_harm_only_max_plus_safe_compliance_control_labeled.jsonl",
    "safe_compliance_control": ROOT
    / "data/responses/d2_ood_harm_stratified_interim/qwen/qwen_d2_ood_safe_compliance_control_labeled.jsonl",
}
ACTIVATION_FILES = [
    ROOT / "data/activations/ood_clean/qwen/ood_advbench_alpaca_clean.pt",
    ROOT / "data/activations/ood_clean/qwen/ood_harmbench_alpaca_clean.pt",
    ROOT / "data/activations/ood_clean/qwen/ood_jailbreakbench_clean.pt",
    ROOT / "data/activations/ood_clean/qwen/ood_xstest_clean.pt",
]
OUT_DIR = ROOT / "data/validation/itw_manual_d2/qwen/harm_stratified_interim"


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
        writer.writerows(rows)


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, -80, 80)
    return 1.0 / (1.0 + np.exp(-values))


def metric_block(y: np.ndarray, proba: np.ndarray) -> dict:
    pred = (proba >= 0.5).astype(int)
    has_two_classes = len(set(y.tolist())) >= 2
    return {
        "n": int(len(y)),
        "refusal": int(y.sum()),
        "compliance": int(len(y) - y.sum()),
        "auroc": float(roc_auc_score(y, proba)) if has_two_classes else math.nan,
        "accuracy": float(accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "mean_p_refusal": float(proba.mean()),
    }


def load_activation_index(paths: list[Path]) -> tuple[dict[str, torch.Tensor], dict]:
    index = {}
    meta = {}
    for path in paths:
        data = torch.load(path, weights_only=False)
        if not meta:
            meta = {
                "n_layers": int(data["n_layers"]),
                "d_model": int(data["d_model"]),
                "pooling": data.get("pooling"),
            }
        for idx, rec_id in enumerate(data["ids"]):
            if rec_id in index:
                raise ValueError(f"duplicate activation id across files: {rec_id}")
            index[rec_id] = data["activations"][idx]
    return index, meta


def stack_activations(rows: list[dict], activation_index: dict[str, torch.Tensor]) -> torch.Tensor:
    missing = [row["id"] for row in rows if row["id"] not in activation_index]
    if missing:
        raise ValueError(f"missing activations for ids: {missing[:20]}")
    return torch.stack([activation_index[row["id"]] for row in rows]).to(torch.float32)


def labels_to_y(rows: list[dict]) -> np.ndarray:
    return np.array([1 if row["behavior_label"] == "refusal" else 0 for row in rows], dtype=int)


def evaluate_layer(layer_payload: dict, x: np.ndarray, y: np.ndarray) -> tuple[dict, np.ndarray]:
    mean = layer_payload["scaler_mean"].to(torch.float32).numpy()
    scale = layer_payload["scaler_scale"].to(torch.float32).numpy()
    coef = layer_payload["coef"].to(torch.float32).numpy()
    intercept = float(layer_payload["intercept"])
    proba = sigmoid(((x - mean) / scale) @ coef + intercept)
    return metric_block(y, proba), proba


def finite_auroc(row: dict) -> float:
    auroc = row["auroc"]
    return auroc if not math.isnan(auroc) else -1.0


def main() -> None:
    probe = torch.load(PROBE, weights_only=False)
    activation_index, activation_meta = load_activation_index(ACTIVATION_FILES)

    layer_rows = []
    prediction_rows = []
    summary = {
        "probe": str(PROBE.relative_to(ROOT)),
        "positive_class": "refusal",
        "negative_class": "compliance",
        "activation_files": [str(path.relative_to(ROOT)) for path in ACTIVATION_FILES],
        "activation_meta": activation_meta,
        "variants": {},
        "note": (
            "This is an interim diagnostic, not a full 2x2 protocol. "
            "The primary variants keep prompt_label fixed to harm."
        ),
    }

    for variant, label_path in LABEL_FILES.items():
        rows = [
            row
            for row in read_jsonl(label_path)
            if row["behavior_label"] in ("refusal", "compliance")
        ]
        y = labels_to_y(rows)
        acts = stack_activations(rows, activation_index)
        proba_by_layer = {}
        variant_layer_rows = []
        for layer_payload in probe["layers"]:
            layer = int(layer_payload["layer"])
            metrics, proba = evaluate_layer(layer_payload, acts[:, layer, :].numpy(), y)
            row = {"variant": variant, "layer": layer, **metrics}
            layer_rows.append(row)
            variant_layer_rows.append(row)
            proba_by_layer[layer] = proba

        has_two_classes = len(set(y.tolist())) >= 2
        best_row = max(variant_layer_rows, key=finite_auroc) if has_two_classes else None
        layer18 = next(row for row in variant_layer_rows if row["layer"] == 18)
        layer20 = next(row for row in variant_layer_rows if row["layer"] == 20)
        summary["variants"][variant] = {
            "label_file": str(label_path.relative_to(ROOT)),
            "counts": {
                "behavior": dict(Counter(row["behavior_label"] for row in rows)),
                "prompt_label": dict(Counter(row["prompt_label"] for row in rows)),
                "target_cell": dict(Counter(row.get("target_cell") for row in rows)),
                "source": dict(Counter(row.get("source") for row in rows)),
            },
            "best_layer_diagnostic": int(best_row["layer"]) if best_row else None,
            "best_layer_metrics_diagnostic": best_row,
            "layer18_ood_balanced_selected_metrics": layer18,
            "layer20_itw_eval_selected_metrics": layer20,
        }

        prediction_layers = sorted(
            {18, 20} | ({int(best_row["layer"])} if best_row else set())
        )
        for idx, row in enumerate(rows):
            pred = {
                "variant": variant,
                "id": row["id"],
                "source": row.get("source"),
                "prompt_label": row.get("prompt_label"),
                "behavior_label": row["behavior_label"],
                "target_cell": row.get("target_cell"),
                "y_refusal": int(y[idx]),
            }
            for layer in prediction_layers:
                pred[f"layer{layer}_p_refusal"] = float(proba_by_layer[layer][idx])
            prediction_rows.append(pred)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    layers_csv = OUT_DIR / "qwen_d2_harm_stratified_interim_layers.csv"
    predictions_csv = OUT_DIR / "qwen_d2_harm_stratified_interim_predictions.csv"
    summary_json = OUT_DIR / "qwen_d2_harm_stratified_interim_summary.json"
    report_md = OUT_DIR / "report.md"
    write_csv(layers_csv, layer_rows, list(layer_rows[0].keys()))
    write_csv(predictions_csv, prediction_rows, sorted({k for row in prediction_rows for k in row}))
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    def fmt(metrics: dict) -> str:
        return (
            f"AUROC `{metrics['auroc']:.4f}`, accuracy `{metrics['accuracy']:.4f}`, "
            f"F1 `{metrics['f1']:.4f}`, mean P(refusal) `{metrics['mean_p_refusal']:.4f}`"
        )

    report = [
        "# Qwen D2 Harm-Stratified Interim Diagnostic",
        "",
        "This is not a full deconfounded 2x2 evaluation because clean `safe|refusal` examples are still unavailable.",
        "The primary diagnostic fixes `prompt_label=harm` and compares observed Qwen behavior:",
        "",
        "- `harm|refusal`",
        "- `harm|compliance`",
        "",
    ]
    for variant in ("source_matched", "harm_only_max", "harm_only_max_plus_safe_compliance_control"):
        item = summary["variants"][variant]
        report.extend(
            [
                f"## {variant}",
                "",
                f"- Counts: `{item['counts']['target_cell']}`",
                f"- Diagnostic best layer: `{item['best_layer_diagnostic']}` with {fmt(item['best_layer_metrics_diagnostic'])}",
                f"- Layer 18 (prior D2 OOD-selected control): {fmt(item['layer18_ood_balanced_selected_metrics'])}",
                f"- Layer 20 (ITW-eval-selected control): {fmt(item['layer20_itw_eval_selected_metrics'])}",
                "",
            ]
        )
    safe_item = summary["variants"]["safe_compliance_control"]
    report.extend(
        [
            "## safe_compliance_control",
            "",
            "This control has only one behavior class, so AUROC is undefined.",
            f"- Layer 18 mean P(refusal): `{safe_item['layer18_ood_balanced_selected_metrics']['mean_p_refusal']:.4f}`",
            f"- Layer 20 mean P(refusal): `{safe_item['layer20_itw_eval_selected_metrics']['mean_p_refusal']:.4f}`",
            "",
        ]
    )
    report_md.write_text("\n".join(report), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"wrote {layers_csv.relative_to(ROOT)}")
    print(f"wrote {predictions_csv.relative_to(ROOT)}")
    print(f"wrote {summary_json.relative_to(ROOT)}")
    print(f"wrote {report_md.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
