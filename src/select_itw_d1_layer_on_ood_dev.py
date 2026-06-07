"""
Select one D1 probe layer on OOD-dev and evaluate it on held-out OOD-test.

The probes themselves are not retrained here. The saved Qwen D1 probe already
contains one independently trained linear probe per layer, trained on ITW train.
This script only chooses a single layer using OOD-dev labels, then reports that
fixed layer on OOD-test.
"""

import argparse
import csv
import json
import sys
from pathlib import Path


ROOT = Path("C:/Users/Vladi/Desktop/projs/interpetability")
DEPS = ROOT / ".deps" / "python"
if DEPS.exists():
    sys.path.insert(0, str(DEPS))

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def labels_to_y(labels) -> np.ndarray:
    return np.array([1 if label == "harm" else 0 for label in labels], dtype=int)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def metrics_from_scores(y_true: np.ndarray, scores: np.ndarray) -> dict:
    pred = (scores >= 0.5).astype(int)
    return {
        "auroc": float(roc_auc_score(y_true, scores)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "mean_p_harm": float(scores.mean()),
    }


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def dataset_name_from_path(path: Path) -> str:
    return path.stem.removesuffix("_clean")


def layer_scores(acts: torch.Tensor, layer_payload: dict, layer: int) -> np.ndarray:
    x = acts[:, layer, :].to(torch.float32).numpy()
    mean = layer_payload["scaler_mean"].numpy()
    scale = layer_payload["scaler_scale"].numpy()
    coef = layer_payload["coef"].numpy()
    intercept = float(layer_payload["intercept"])
    logits = ((x - mean) / scale) @ coef + intercept
    return sigmoid(logits)


def stratified_dev_test_indices(y: np.ndarray, dev_frac: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    dev_parts = []
    test_parts = []
    for label in sorted(set(y.tolist())):
        idx = np.flatnonzero(y == label)
        rng.shuffle(idx)
        n_dev = int(round(len(idx) * dev_frac))
        n_dev = min(max(n_dev, 1), len(idx) - 1)
        dev_parts.append(idx[:n_dev])
        test_parts.append(idx[n_dev:])
    dev_idx = np.sort(np.concatenate(dev_parts))
    test_idx = np.sort(np.concatenate(test_parts))
    return dev_idx, test_idx


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_text_baselines(text_baseline_dir: Path, dataset: str) -> dict[str, dict]:
    path = text_baseline_dir / dataset / "predictions.csv"
    if not path.exists():
        return {}
    return {row["id"]: row for row in read_csv(path)}


def score_text_baselines(ids: list[str], y: np.ndarray, text_rows: dict[str, dict]) -> dict:
    if not text_rows:
        return {}
    tfidf_scores = np.array([float(text_rows[rec_id]["tfidf_p_harm"]) for rec_id in ids], dtype=float)
    length_scores = np.array([float(text_rows[rec_id]["length_logreg_p_harm"]) for rec_id in ids], dtype=float)
    char_lens = np.array([float(text_rows[rec_id]["char_len"]) for rec_id in ids], dtype=float)
    token_counts = np.array([float(text_rows[rec_id]["qwen_token_count"]) for rec_id in ids], dtype=float)
    return {
        "tfidf_logreg": metrics_from_scores(y, tfidf_scores),
        "length_token_logreg": metrics_from_scores(y, length_scores),
        "char_len_as_harm_score": metrics_from_scores(y, char_lens),
        "qwen_token_count_as_harm_score": metrics_from_scores(y, token_counts),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", default="data/probes/itw_manual_d1/qwen.pt")
    parser.add_argument("--activation-dir", default="data/activations/ood_clean/qwen")
    parser.add_argument("--text-baseline-dir", default="data/validation/itw_manual_d1/qwen/ood_clean/text_baselines")
    parser.add_argument("--out-dir", default="data/validation/itw_manual_d1/qwen/ood_clean/ood_dev_test_layer_selection")
    parser.add_argument("--dev-frac", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    probe_payload = torch.load(ROOT / args.probe, weights_only=False)
    activation_paths = sorted((ROOT / args.activation_dir).glob("*.pt"))
    if not activation_paths:
        raise FileNotFoundError(ROOT / args.activation_dir)

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    text_baseline_dir = ROOT / args.text_baseline_dir

    datasets = []
    for ds_idx, path in enumerate(activation_paths):
        data = torch.load(path, weights_only=False)
        y = labels_to_y(data["labels"])
        rng = np.random.default_rng(args.seed + ds_idx)
        dev_idx, test_idx = stratified_dev_test_indices(y, args.dev_frac, rng)
        datasets.append(
            {
                "dataset": dataset_name_from_path(path),
                "path": path,
                "data": data,
                "y": y,
                "dev_idx": dev_idx,
                "test_idx": test_idx,
            }
        )

    n_layers = int(datasets[0]["data"]["n_layers"])
    dev_rows = []
    test_rows = []
    for ds in datasets:
        acts = ds["data"]["activations"]
        y = ds["y"]
        for layer in range(n_layers):
            scores = layer_scores(acts, probe_payload["layers"][layer], layer)
            for split, idx, target_rows in [
                ("dev", ds["dev_idx"], dev_rows),
                ("test", ds["test_idx"], test_rows),
            ]:
                split_metrics = metrics_from_scores(y[idx], scores[idx])
                target_rows.append(
                    {
                        "dataset": ds["dataset"],
                        "split": split,
                        "layer": layer,
                        "n_records": int(len(idx)),
                        "n_harm": int(y[idx].sum()),
                        "n_safe": int(len(idx) - y[idx].sum()),
                        **split_metrics,
                    }
                )

    layer_selection_rows = []
    for layer in range(n_layers):
        layer_dev = [row for row in dev_rows if row["layer"] == layer]
        layer_test = [row for row in test_rows if row["layer"] == layer]
        layer_selection_rows.append(
            {
                "layer": layer,
                "mean_dev_auroc": float(np.mean([row["auroc"] for row in layer_dev])),
                "mean_test_auroc": float(np.mean([row["auroc"] for row in layer_test])),
                "mean_dev_accuracy": float(np.mean([row["accuracy"] for row in layer_dev])),
                "mean_test_accuracy": float(np.mean([row["accuracy"] for row in layer_test])),
                "mean_dev_f1": float(np.mean([row["f1"] for row in layer_dev])),
                "mean_test_f1": float(np.mean([row["f1"] for row in layer_test])),
            }
        )
    selected = max(layer_selection_rows, key=lambda row: row["mean_dev_auroc"])
    selected_layer = int(selected["layer"])

    selected_test_rows = [row for row in test_rows if row["layer"] == selected_layer]
    selected_dev_rows = [row for row in dev_rows if row["layer"] == selected_layer]

    text_baseline_rows = []
    for ds in datasets:
        text_rows = load_text_baselines(text_baseline_dir, ds["dataset"])
        for split, idx in [("dev", ds["dev_idx"]), ("test", ds["test_idx"])]:
            ids = [ds["data"]["ids"][int(i)] for i in idx]
            y_split = ds["y"][idx]
            baseline_metrics = score_text_baselines(ids, y_split, text_rows)
            if not baseline_metrics:
                continue
            text_baseline_rows.append(
                {
                    "dataset": ds["dataset"],
                    "split": split,
                    "n_records": int(len(idx)),
                    "n_harm": int(y_split.sum()),
                    "n_safe": int(len(idx) - y_split.sum()),
                    "tfidf_auroc": baseline_metrics["tfidf_logreg"]["auroc"],
                    "length_token_auroc": baseline_metrics["length_token_logreg"]["auroc"],
                    "char_len_auroc": baseline_metrics["char_len_as_harm_score"]["auroc"],
                    "qwen_token_count_auroc": baseline_metrics["qwen_token_count_as_harm_score"]["auroc"],
                }
            )

    write_csv(
        out_dir / "dev_layer_metrics.csv",
        dev_rows,
        ["dataset", "split", "layer", "n_records", "n_harm", "n_safe", "auroc", "accuracy", "f1", "precision", "recall", "mean_p_harm"],
    )
    write_csv(
        out_dir / "test_layer_metrics.csv",
        test_rows,
        ["dataset", "split", "layer", "n_records", "n_harm", "n_safe", "auroc", "accuracy", "f1", "precision", "recall", "mean_p_harm"],
    )
    write_csv(
        out_dir / "layer_selection_summary.csv",
        layer_selection_rows,
        ["layer", "mean_dev_auroc", "mean_test_auroc", "mean_dev_accuracy", "mean_test_accuracy", "mean_dev_f1", "mean_test_f1"],
    )
    write_csv(
        out_dir / "selected_layer_dev_metrics.csv",
        selected_dev_rows,
        ["dataset", "split", "layer", "n_records", "n_harm", "n_safe", "auroc", "accuracy", "f1", "precision", "recall", "mean_p_harm"],
    )
    write_csv(
        out_dir / "selected_layer_test_metrics.csv",
        selected_test_rows,
        ["dataset", "split", "layer", "n_records", "n_harm", "n_safe", "auroc", "accuracy", "f1", "precision", "recall", "mean_p_harm"],
    )
    if text_baseline_rows:
        write_csv(
            out_dir / "text_baseline_dev_test_metrics.csv",
            text_baseline_rows,
            ["dataset", "split", "n_records", "n_harm", "n_safe", "tfidf_auroc", "length_token_auroc", "char_len_auroc", "qwen_token_count_auroc"],
        )

    payload = {
        "probe": args.probe,
        "activation_dir": args.activation_dir,
        "text_baseline_dir": args.text_baseline_dir,
        "dev_frac": args.dev_frac,
        "seed": args.seed,
        "selection_rule": "single global layer maximizing unweighted mean AUROC across OOD-dev datasets",
        "selected_layer": selected_layer,
        "selected_layer_mean_dev_auroc": selected["mean_dev_auroc"],
        "selected_layer_mean_test_auroc": selected["mean_test_auroc"],
        "datasets": [
            {
                "dataset": ds["dataset"],
                "n_records": int(len(ds["y"])),
                "dev_records": int(len(ds["dev_idx"])),
                "test_records": int(len(ds["test_idx"])),
                "dev_harm": int(ds["y"][ds["dev_idx"]].sum()),
                "test_harm": int(ds["y"][ds["test_idx"]].sum()),
            }
            for ds in datasets
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    report = [
        "# OOD-Dev Layer Selection and OOD-Test Evaluation",
        "",
        "The probes are still trained only on the ITW train split. OOD-dev is used",
        "only to choose one global layer. OOD-test is then used for the held-out",
        "evaluation of that fixed layer.",
        "",
        f"- Selection rule: `{payload['selection_rule']}`",
        f"- Dev fraction per OOD dataset: `{args.dev_frac}`",
        f"- Seed: `{args.seed}`",
        f"- Selected layer: `{selected_layer}`",
        f"- Mean OOD-dev AUROC at selected layer: `{selected['mean_dev_auroc']:.4f}`",
        f"- Mean OOD-test AUROC at selected layer: `{selected['mean_test_auroc']:.4f}`",
        "",
        "## Selected Layer Test Metrics",
        "",
        "| Dataset | Test n | Test harm | AUROC | Accuracy | F1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in selected_test_rows:
        report.append(
            f"| `{row['dataset']}` | {row['n_records']} | {row['n_harm']} | "
            f"{row['auroc']:.4f} | {row['accuracy']:.4f} | {row['f1']:.4f} |"
        )

    if text_baseline_rows:
        report.extend(
            [
                "",
                "## OOD-Test Text Baselines",
                "",
                "| Dataset | TF-IDF AUROC | Length/token AUROC | Qwen-token AUROC |",
                "|---|---:|---:|---:|",
            ]
        )
        for row in text_baseline_rows:
            if row["split"] != "test":
                continue
            report.append(
                f"| `{row['dataset']}` | {row['tfidf_auroc']:.4f} | "
                f"{row['length_token_auroc']:.4f} | {row['qwen_token_count_auroc']:.4f} |"
            )

    report.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This is stricter than selecting the best layer on each full OOD dataset.",
            "- It is still OOD-informed model selection, so final claims should call it",
            "  `OOD-dev selected`, not `ITW-selected`.",
            "- ITW eval is not used in this layer choice; it remains an in-domain",
            "  diagnostic and historical comparison point.",
            "",
        ]
    )
    (out_dir / "report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
