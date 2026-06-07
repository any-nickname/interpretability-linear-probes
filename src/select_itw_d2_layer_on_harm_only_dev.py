"""
Select a Qwen D2 layer on harm-only OOD-dev and evaluate on harm-only OOD-test.

This is an interim D2 layer-selection protocol for the current project state,
where clean `safe|refusal` rows are still unavailable. It fixes the prompt-risk
label to `harm` and selects a layer for predicting observed Qwen behavior:

- positive: refusal
- negative: compliance

The script evaluates two variants:

- `source_matched`: smaller, source-balanced harm-only rows.
- `harm_only_max`: larger, all confirmed harm|compliance rows plus matched
  harm|refusal rows.
"""

import csv
import json
import math
import random
import sys
from collections import Counter, defaultdict
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
}
ACTIVATION_FILES = [
    ROOT / "data/activations/ood_clean/qwen/ood_advbench_alpaca_clean.pt",
    ROOT / "data/activations/ood_clean/qwen/ood_harmbench_alpaca_clean.pt",
    ROOT / "data/activations/ood_clean/qwen/ood_jailbreakbench_clean.pt",
    ROOT / "data/activations/ood_clean/qwen/ood_xstest_clean.pt",
]
OUT_DIR = ROOT / "data/validation/itw_manual_d2/qwen/harm_only_dev_test_layer_selection"
SPLIT_OUT_DIR = ROOT / "data/responses/d2_ood_harm_stratified_interim/qwen"


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


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


def metric_block(y: np.ndarray, proba: np.ndarray) -> dict:
    pred = (proba >= 0.5).astype(int)
    return {
        "n": int(len(y)),
        "refusal": int(y.sum()),
        "compliance": int(len(y) - y.sum()),
        "auroc": float(roc_auc_score(y, proba)) if len(set(y.tolist())) >= 2 else math.nan,
        "accuracy": float(accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "mean_p_refusal": float(proba.mean()),
    }


def split_source_behavior(rows: list[dict], seed: int = 42) -> tuple[list[dict], list[dict], dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[(row["source"], row["behavior_label"])].append(row)

    rng = random.Random(seed)
    dev = []
    test = []
    group_summary = {}
    for key in sorted(groups):
        group = sorted(groups[key], key=lambda row: row["id"])
        rng.shuffle(group)
        n_dev = max(1, int(round(len(group) * 0.5)))
        dev_part = group[:n_dev]
        test_part = group[n_dev:]
        dev.extend(dev_part)
        test.extend(test_part)
        group_summary["|".join(key)] = {
            "total": len(group),
            "dev": len(dev_part),
            "test": len(test_part),
        }

    dev = sorted(dev, key=lambda row: (row["behavior_label"], row["source"], row["id"]))
    test = sorted(test, key=lambda row: (row["behavior_label"], row["source"], row["id"]))
    if len({row["behavior_label"] for row in dev}) < 2:
        raise ValueError("dev split lacks both behavior classes")
    if len({row["behavior_label"] for row in test}) < 2:
        raise ValueError("test split lacks both behavior classes")
    return dev, test, group_summary


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


def labels_to_y(rows: list[dict]) -> np.ndarray:
    return np.array([1 if row["behavior_label"] == "refusal" else 0 for row in rows], dtype=int)


def evaluate_layer(layer_payload: dict, acts: torch.Tensor, layer: int, y: np.ndarray) -> tuple[dict, np.ndarray]:
    x = acts[:, layer, :].to(torch.float32).numpy()
    mean = layer_payload["scaler_mean"].to(torch.float32).numpy()
    scale = layer_payload["scaler_scale"].to(torch.float32).numpy()
    coef = layer_payload["coef"].to(torch.float32).numpy()
    intercept = float(layer_payload["intercept"])
    proba = sigmoid(((x - mean) / scale) @ coef + intercept)
    return metric_block(y, proba), proba


def summarize_rows(rows: list[dict]) -> dict:
    return {
        "n": len(rows),
        "behavior": dict(Counter(row["behavior_label"] for row in rows)),
        "source": dict(Counter(row["source"] for row in rows)),
        "source_behavior": dict(
            sorted(Counter(f"{row['source']}|{row['behavior_label']}" for row in rows).items())
        ),
    }


def main() -> None:
    probe = torch.load(PROBE, weights_only=False)
    activation_index = load_activation_index(ACTIVATION_FILES)

    all_layer_rows = []
    all_prediction_rows = []
    summary = {
        "probe": str(PROBE.relative_to(ROOT)),
        "positive_class": "refusal",
        "negative_class": "compliance",
        "split_strategy": "stratified by source and behavior_label with seed=42; singleton source-behavior groups go to dev",
        "activation_files": [str(path.relative_to(ROOT)) for path in ACTIVATION_FILES],
        "variants": {},
        "note": (
            "This is an interim D2 layer-selection protocol. It fixes prompt_label=harm "
            "and does not replace a future full 2x2 protocol with clean safe|refusal rows."
        ),
    }

    for variant, label_path in LABEL_FILES.items():
        rows = [
            row
            for row in read_jsonl(label_path)
            if row["prompt_label"] == "harm"
            and row["behavior_label"] in ("refusal", "compliance")
        ]
        dev_rows, test_rows, split_group_summary = split_source_behavior(rows)
        for split_name, split_rows in (("dev", dev_rows), ("test", test_rows)):
            tagged = []
            for row in split_rows:
                out = dict(row)
                out["harm_only_split"] = split_name
                out["harm_only_split_variant"] = variant
                tagged.append(out)
            path = SPLIT_OUT_DIR / f"qwen_d2_ood_harm_only_{variant}_{split_name}_labeled.jsonl"
            write_jsonl(path, tagged)

        y_dev = labels_to_y(dev_rows)
        y_test = labels_to_y(test_rows)
        acts_dev = stack_activations(dev_rows, activation_index)
        acts_test = stack_activations(test_rows, activation_index)

        variant_layer_rows = []
        dev_proba_by_layer = {}
        test_proba_by_layer = {}
        for layer_payload in probe["layers"]:
            layer = int(layer_payload["layer"])
            dev_metrics, dev_proba = evaluate_layer(layer_payload, acts_dev, layer, y_dev)
            test_metrics, test_proba = evaluate_layer(layer_payload, acts_test, layer, y_test)
            row = {
                "variant": variant,
                "layer": layer,
                **{f"dev_{key}": value for key, value in dev_metrics.items()},
                **{f"test_{key}": value for key, value in test_metrics.items()},
            }
            variant_layer_rows.append(row)
            all_layer_rows.append(row)
            dev_proba_by_layer[layer] = dev_proba
            test_proba_by_layer[layer] = test_proba

        selected = max(variant_layer_rows, key=lambda row: row["dev_auroc"])
        selected_layer = int(selected["layer"])
        layer18 = next(row for row in variant_layer_rows if int(row["layer"]) == 18)
        layer20 = next(row for row in variant_layer_rows if int(row["layer"]) == 20)
        best_test = max(variant_layer_rows, key=lambda row: row["test_auroc"])

        for split_name, split_rows, y, proba_by_layer in (
            ("dev", dev_rows, y_dev, dev_proba_by_layer),
            ("test", test_rows, y_test, test_proba_by_layer),
        ):
            for idx, row in enumerate(split_rows):
                all_prediction_rows.append(
                    {
                        "variant": variant,
                        "split": split_name,
                        "id": row["id"],
                        "source": row.get("source"),
                        "behavior_label": row["behavior_label"],
                        "target_cell": row.get("target_cell"),
                        "y_refusal": int(y[idx]),
                        f"layer{selected_layer}_p_refusal": float(proba_by_layer[selected_layer][idx]),
                        "layer18_p_refusal": float(proba_by_layer[18][idx]),
                        "layer20_p_refusal": float(proba_by_layer[20][idx]),
                    }
                )

        summary["variants"][variant] = {
            "label_file": str(label_path.relative_to(ROOT)),
            "split_group_summary": split_group_summary,
            "dev": summarize_rows(dev_rows),
            "test": summarize_rows(test_rows),
            "selected_layer_by_dev_auroc": selected_layer,
            "selected_layer_dev_metrics": {
                key.removeprefix("dev_"): value for key, value in selected.items() if key.startswith("dev_")
            },
            "selected_layer_test_metrics": {
                key.removeprefix("test_"): value for key, value in selected.items() if key.startswith("test_")
            },
            "layer18_control_dev_metrics": {
                key.removeprefix("dev_"): value for key, value in layer18.items() if key.startswith("dev_")
            },
            "layer18_control_test_metrics": {
                key.removeprefix("test_"): value for key, value in layer18.items() if key.startswith("test_")
            },
            "layer20_control_dev_metrics": {
                key.removeprefix("dev_"): value for key, value in layer20.items() if key.startswith("dev_")
            },
            "layer20_control_test_metrics": {
                key.removeprefix("test_"): value for key, value in layer20.items() if key.startswith("test_")
            },
            "best_test_layer_exploratory": int(best_test["layer"]),
            "best_test_layer_metrics_exploratory": {
                key.removeprefix("test_"): value for key, value in best_test.items() if key.startswith("test_")
            },
        }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    layer_csv = OUT_DIR / "qwen_d2_harm_only_dev_test_layers.csv"
    prediction_csv = OUT_DIR / "qwen_d2_harm_only_dev_test_predictions.csv"
    summary_json = OUT_DIR / "qwen_d2_harm_only_dev_test_summary.json"
    report_md = OUT_DIR / "report.md"
    write_csv(layer_csv, all_layer_rows, list(all_layer_rows[0].keys()))
    write_csv(prediction_csv, all_prediction_rows, sorted({key for row in all_prediction_rows for key in row}))
    summary["outputs"] = {
        "layers": str(layer_csv.relative_to(ROOT)),
        "predictions": str(prediction_csv.relative_to(ROOT)),
        "summary": str(summary_json.relative_to(ROOT)),
        "report": str(report_md.relative_to(ROOT)),
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    def fmt(metrics: dict) -> str:
        return f"AUROC `{metrics['auroc']:.4f}`, accuracy `{metrics['accuracy']:.4f}`, F1 `{metrics['f1']:.4f}`"

    report = [
        "# Qwen D2 Harm-Only Dev/Test Layer Selection",
        "",
        "This is an interim protocol while clean `safe|refusal` rows are unavailable.",
        "The prompt-risk label is fixed to `harm`; the target is observed Qwen behavior.",
        "",
    ]
    for variant, item in summary["variants"].items():
        report.extend(
            [
                f"## {variant}",
                "",
                f"- Dev counts: `{item['dev']['source_behavior']}`",
                f"- Test counts: `{item['test']['source_behavior']}`",
                f"- Selected layer by dev AUROC: `{item['selected_layer_by_dev_auroc']}`",
                f"- Selected layer dev: {fmt(item['selected_layer_dev_metrics'])}",
                f"- Selected layer test: {fmt(item['selected_layer_test_metrics'])}",
                f"- Layer 18 control test: {fmt(item['layer18_control_test_metrics'])}",
                f"- Layer 20 control test: {fmt(item['layer20_control_test_metrics'])}",
                f"- Exploratory best test layer: `{item['best_test_layer_exploratory']}` with {fmt(item['best_test_layer_metrics_exploratory'])}",
                "",
            ]
        )
    report_md.write_text("\n".join(report), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"wrote {report_md.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
