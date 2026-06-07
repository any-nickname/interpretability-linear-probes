"""
Select a Qwen D2 layer on manually labeled OOD-dev and evaluate on OOD-test.

The ITW-trained D2 probes are kept fixed. This script only chooses which layer
to report using OOD-dev AUROC, then evaluates that fixed layer on OOD-test.
"""

import argparse
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
    metrics = {
        "n": int(len(y)),
        "refusal": int(y.sum()),
        "compliance": int(len(y) - y.sum()),
        "accuracy": float(accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "mean_p_refusal": float(proba.mean()),
    }
    metrics["auroc"] = (
        float(roc_auc_score(y, proba)) if len(set(y.tolist())) >= 2 else math.nan
    )
    return metrics


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


def labels_to_y(rows: list[dict]) -> np.ndarray:
    return np.array([1 if row["behavior_label"] == "refusal" else 0 for row in rows], dtype=int)


def stack_activations(rows: list[dict], activation_index: dict[str, torch.Tensor]) -> torch.Tensor:
    missing = [row["id"] for row in rows if row["id"] not in activation_index]
    if missing:
        raise ValueError(f"missing activations for ids: {missing[:10]}")
    return torch.stack([activation_index[row["id"]] for row in rows]).to(torch.float32)


def evaluate_layer(layer_payload: dict, x: np.ndarray, y: np.ndarray) -> tuple[dict, np.ndarray]:
    mean = layer_payload["scaler_mean"].to(torch.float32).numpy()
    scale = layer_payload["scaler_scale"].to(torch.float32).numpy()
    coef = layer_payload["coef"].to(torch.float32).numpy()
    intercept = float(layer_payload["intercept"])
    xs = (x - mean) / scale
    proba = sigmoid(xs @ coef + intercept)
    return metric_block(y, proba), proba


def group_metrics(
    rows: list[dict], y: np.ndarray, proba: np.ndarray, layer: int, split: str
) -> list[dict]:
    out = []
    for group_key in ("source", "prompt_label", "manual_label_source"):
        for value in sorted({row.get(group_key) for row in rows}):
            idxs = [idx for idx, row in enumerate(rows) if row.get(group_key) == value]
            yy = y[idxs]
            pp = proba[idxs]
            out.append(
                {
                    "split": split,
                    "group_key": group_key,
                    "group_value": value,
                    "layer": layer,
                    **metric_block(yy, pp),
                }
            )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", default="data/probes/itw_manual_d2/qwen.pt")
    parser.add_argument(
        "--dev-labels",
        default="data/responses/d2_ood_balanced_200/qwen/qwen_d2_ood_balanced_200_dev_labeled.jsonl",
    )
    parser.add_argument(
        "--test-labels",
        default="data/responses/d2_ood_balanced_200/qwen/qwen_d2_ood_balanced_200_test_labeled.jsonl",
    )
    parser.add_argument(
        "--activation-files",
        nargs="+",
        default=[
            "data/activations/ood_clean/qwen/ood_advbench_alpaca_clean.pt",
            "data/activations/ood_clean/qwen/ood_harmbench_alpaca_clean.pt",
            "data/activations/ood_clean/qwen/ood_jailbreakbench_clean.pt",
            "data/activations/ood_clean/qwen/ood_xstest_clean.pt",
        ],
    )
    parser.add_argument("--itw-selected-layer", type=int, default=20)
    parser.add_argument(
        "--out-dir",
        default="data/validation/itw_manual_d2/qwen/ood_balanced_200/layer_selection",
    )
    args = parser.parse_args()

    probe = torch.load(ROOT / args.probe, weights_only=False)
    dev_rows = [
        row for row in read_jsonl(ROOT / args.dev_labels) if row["behavior_label"] in ("refusal", "compliance")
    ]
    test_rows = [
        row for row in read_jsonl(ROOT / args.test_labels) if row["behavior_label"] in ("refusal", "compliance")
    ]
    y_dev = labels_to_y(dev_rows)
    y_test = labels_to_y(test_rows)
    if len(set(y_dev.tolist())) < 2 or len(set(y_test.tolist())) < 2:
        raise ValueError("dev/test must each contain both behavior classes")

    activation_index, activation_meta = load_activation_index([ROOT / p for p in args.activation_files])
    acts_dev = stack_activations(dev_rows, activation_index)
    acts_test = stack_activations(test_rows, activation_index)

    layer_rows = []
    proba_by_layer_dev = {}
    proba_by_layer_test = {}
    for layer_payload in probe["layers"]:
        layer = int(layer_payload["layer"])
        dev_metrics, proba_dev = evaluate_layer(
            layer_payload, acts_dev[:, layer, :].numpy(), y_dev
        )
        test_metrics, proba_test = evaluate_layer(
            layer_payload, acts_test[:, layer, :].numpy(), y_test
        )
        proba_by_layer_dev[layer] = proba_dev
        proba_by_layer_test[layer] = proba_test
        layer_rows.append(
            {
                "layer": layer,
                **{f"dev_{k}": v for k, v in dev_metrics.items()},
                **{f"test_{k}": v for k, v in test_metrics.items()},
            }
        )

    selected = max(layer_rows, key=lambda row: row["dev_auroc"])
    selected_layer = int(selected["layer"])
    itw_control = next(row for row in layer_rows if int(row["layer"]) == args.itw_selected_layer)
    best_test_exploratory = max(layer_rows, key=lambda row: row["test_auroc"])

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    layers_csv = out_dir / "qwen_d2_ood_balanced_200_layers.csv"
    write_csv(layers_csv, layer_rows, list(layer_rows[0].keys()))

    prediction_rows = []
    for split, rows, y, proba_map in (
        ("ood_dev", dev_rows, y_dev, proba_by_layer_dev),
        ("ood_test", test_rows, y_test, proba_by_layer_test),
    ):
        for idx, row in enumerate(rows):
            prediction_rows.append(
                {
                    "split": split,
                    "id": row["id"],
                    "source": row.get("source"),
                    "prompt_label": row.get("prompt_label"),
                    "behavior_label": row["behavior_label"],
                    "manual_label_source": row.get("manual_label_source"),
                    "y_refusal": int(y[idx]),
                    f"layer{selected_layer}_p_refusal": float(proba_map[selected_layer][idx]),
                    f"layer{args.itw_selected_layer}_p_refusal": float(
                        proba_map[args.itw_selected_layer][idx]
                    ),
                }
            )
    predictions_csv = out_dir / "qwen_d2_ood_balanced_200_predictions.csv"
    write_csv(predictions_csv, prediction_rows, list(prediction_rows[0].keys()))

    group_rows = []
    group_rows.extend(
        group_metrics(
            dev_rows,
            y_dev,
            proba_by_layer_dev[selected_layer],
            selected_layer,
            "ood_dev",
        )
    )
    group_rows.extend(
        group_metrics(
            test_rows,
            y_test,
            proba_by_layer_test[selected_layer],
            selected_layer,
            "ood_test",
        )
    )
    groups_csv = out_dir / "qwen_d2_ood_balanced_200_selected_layer_groups.csv"
    write_csv(groups_csv, group_rows, list(group_rows[0].keys()))

    summary = {
        "probe": args.probe,
        "dev_labels": args.dev_labels,
        "test_labels": args.test_labels,
        "activation_files": args.activation_files,
        "positive_class": "refusal",
        "negative_class": "compliance",
        "dev_counts": dict(Counter(row["behavior_label"] for row in dev_rows)),
        "test_counts": dict(Counter(row["behavior_label"] for row in test_rows)),
        "selected_layer_by_ood_dev": selected_layer,
        "selected_layer_dev_metrics": {
            key.removeprefix("dev_"): value for key, value in selected.items() if key.startswith("dev_")
        },
        "selected_layer_test_metrics": {
            key.removeprefix("test_"): value for key, value in selected.items() if key.startswith("test_")
        },
        "itw_eval_selected_layer_control": args.itw_selected_layer,
        "itw_eval_selected_layer_control_dev_metrics": {
            key.removeprefix("dev_"): value for key, value in itw_control.items() if key.startswith("dev_")
        },
        "itw_eval_selected_layer_control_test_metrics": {
            key.removeprefix("test_"): value for key, value in itw_control.items() if key.startswith("test_")
        },
        "best_test_layer_exploratory": int(best_test_exploratory["layer"]),
        "best_test_layer_metrics_exploratory": {
            key.removeprefix("test_"): value
            for key, value in best_test_exploratory.items()
            if key.startswith("test_")
        },
        "activation_meta": activation_meta,
        "note": "The selected layer is chosen only on OOD-dev. The best test layer is diagnostic only.",
    }
    summary_path = out_dir / "qwen_d2_ood_balanced_200_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report = [
        "# Qwen D2 OOD-Dev/Test Layer Selection",
        "",
        "- Probe: ITW-trained Qwen D2 response-behavior probe",
        "- Positive class: `refusal`",
        "- Negative class: `compliance`",
        "- OOD-dev: `50` refusal / `50` compliance",
        "- OOD-test: `50` refusal / `50` compliance",
        f"- Selected layer by OOD-dev AUROC: `{selected_layer}`",
        f"- Selected layer OOD-dev AUROC/accuracy/F1: `{summary['selected_layer_dev_metrics']['auroc']:.4f}` / `{summary['selected_layer_dev_metrics']['accuracy']:.4f}` / `{summary['selected_layer_dev_metrics']['f1']:.4f}`",
        f"- Selected layer OOD-test AUROC/accuracy/F1: `{summary['selected_layer_test_metrics']['auroc']:.4f}` / `{summary['selected_layer_test_metrics']['accuracy']:.4f}` / `{summary['selected_layer_test_metrics']['f1']:.4f}`",
        f"- ITW-eval-selected layer `{args.itw_selected_layer}` OOD-test AUROC: `{summary['itw_eval_selected_layer_control_test_metrics']['auroc']:.4f}`",
        f"- Exploratory best OOD-test layer: `{summary['best_test_layer_exploratory']}` with AUROC `{summary['best_test_layer_metrics_exploratory']['auroc']:.4f}`",
        "",
        "The exploratory best OOD-test layer is not used for selection.",
        "",
    ]
    report_path = out_dir / "report.md"
    report_path.write_text("\n".join(report), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"wrote {layers_csv.relative_to(ROOT)}")
    print(f"wrote {predictions_csv.relative_to(ROOT)}")
    print(f"wrote {groups_csv.relative_to(ROOT)}")
    print(f"wrote {report_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
