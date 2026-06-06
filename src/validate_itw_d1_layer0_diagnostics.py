"""
Layer-0 diagnostics for the manual In-the-Wild Qwen D1 probe.

This script scores eval examples with the saved activation probe and joins the
scores with exact-split text baselines. The goal is not to prove semantics, but
to inspect whether high layer-0 confidence tracks surface baselines or obvious
prompt artifacts.
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path


ROOT = Path("C:/Users/Vladi/Desktop/projs/interpetability")
DEPS = ROOT / ".deps" / "python"
if DEPS.exists():
    sys.path.insert(0, str(DEPS))

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def labels_to_y(labels) -> np.ndarray:
    return np.array([1 if label == "harm" else 0 for label in labels], dtype=int)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def pearson(xs, ys) -> float | None:
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    if len(xs) < 2 or np.std(xs) == 0 or np.std(ys) == 0:
        return None
    return float(np.corrcoef(xs, ys)[0, 1])


def rankdata(values) -> np.ndarray:
    pairs = sorted((float(value), idx) for idx, value in enumerate(values))
    ranks = np.zeros(len(pairs), dtype=float)
    i = 0
    while i < len(pairs):
        j = i + 1
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for _, idx in pairs[i:j]:
            ranks[idx] = avg_rank
        i = j
    return ranks


def spearman(xs, ys) -> float | None:
    return pearson(rankdata(xs), rankdata(ys))


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def layer_scores(eval_data: dict, probe_payload: dict, layer: int) -> np.ndarray:
    layer_payload = probe_payload["layers"][layer]
    acts = eval_data["activations"][:, layer, :].to(torch.float32).numpy()
    mean = layer_payload["scaler_mean"].numpy()
    scale = layer_payload["scaler_scale"].numpy()
    coef = layer_payload["coef"].numpy()
    intercept = float(layer_payload["intercept"])
    scaled = (acts - mean) / scale
    logits = scaled @ coef + intercept
    return sigmoid(logits)


def metrics(y_true: np.ndarray, scores: np.ndarray) -> dict:
    pred = (scores >= 0.5).astype(int)
    return {
        "auroc": float(roc_auc_score(y_true, scores)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-activations", default="data/activations/qwen/itw_manual_d1_eval.pt")
    parser.add_argument("--probe", default="data/probes/itw_manual_d1/qwen.pt")
    parser.add_argument("--baseline-predictions", default="data/validation/itw_manual_d1/qwen/text_baselines/eval_predictions.csv")
    parser.add_argument("--out-dir", default="data/validation/itw_manual_d1/qwen/layer0_diagnostics")
    parser.add_argument("--compare-layer", type=int, default=12)
    args = parser.parse_args()

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_data = torch.load(ROOT / args.eval_activations, weights_only=False)
    probe_payload = torch.load(ROOT / args.probe, weights_only=False)
    baseline_rows = {row["id"]: row for row in read_csv(ROOT / args.baseline_predictions)}

    y_true = labels_to_y(eval_data["labels"])
    layer0 = layer_scores(eval_data, probe_payload, 0)
    compare = layer_scores(eval_data, probe_payload, args.compare_layer)

    rows = []
    for idx, rec_id in enumerate(eval_data["ids"]):
        base = baseline_rows.get(rec_id, {})
        y = int(y_true[idx])
        score0 = float(layer0[idx])
        score_cmp = float(compare[idx])
        pred0 = 1 if score0 >= 0.5 else 0
        prompt = eval_data["prompts"][idx]
        row = {
            "id": rec_id,
            "pair_id": rec_id.rsplit("_", 1)[0],
            "label": eval_data["labels"][idx],
            "y": y,
            "layer0_p_harm": score0,
            f"layer{args.compare_layer}_p_harm": score_cmp,
            "layer0_pred": "harm" if pred0 == 1 else "safe",
            "layer0_correct": pred0 == y,
            "layer0_confidence": abs(score0 - 0.5),
            "tfidf_p_harm": float(base["tfidf_p_harm"]) if base else "",
            "length_logreg_p_harm": float(base["length_logreg_p_harm"]) if base else "",
            "char_len": int(base["char_len"]) if base else len(prompt),
            "word_len": int(base["word_len"]) if base else len(prompt.split()),
            "qwen_token_count": int(base["qwen_token_count"]) if base else "",
            "prompt_preview": " ".join(prompt.split())[:360],
        }
        rows.append(row)

    fieldnames = [
        "id",
        "pair_id",
        "label",
        "y",
        "layer0_p_harm",
        f"layer{args.compare_layer}_p_harm",
        "layer0_pred",
        "layer0_correct",
        "layer0_confidence",
        "tfidf_p_harm",
        "length_logreg_p_harm",
        "char_len",
        "word_len",
        "qwen_token_count",
        "prompt_preview",
    ]
    write_csv(out_dir / "layer0_eval_scores.csv", rows, fieldnames)
    write_csv(out_dir / "layer0_top_harm_like.csv", sorted(rows, key=lambda r: -r["layer0_p_harm"])[:20], fieldnames)
    write_csv(out_dir / "layer0_top_safe_like.csv", sorted(rows, key=lambda r: r["layer0_p_harm"])[:20], fieldnames)
    errors = [row for row in rows if not row["layer0_correct"]]
    write_csv(out_dir / "layer0_confident_errors.csv", sorted(errors, key=lambda r: -r["layer0_confidence"]), fieldnames)

    tfidf_scores = [float(row["tfidf_p_harm"]) for row in rows if row["tfidf_p_harm"] != ""]
    length_scores = [float(row["length_logreg_p_harm"]) for row in rows if row["length_logreg_p_harm"] != ""]
    char_lens = [float(row["char_len"]) for row in rows]
    token_counts = [float(row["qwen_token_count"]) for row in rows if row["qwen_token_count"] != ""]
    layer0_for_baselines = [row["layer0_p_harm"] for row in rows if row["tfidf_p_harm"] != ""]
    layer0_for_tokens = [row["layer0_p_harm"] for row in rows if row["qwen_token_count"] != ""]

    summary = {
        "eval_activations": args.eval_activations,
        "probe": args.probe,
        "baseline_predictions": args.baseline_predictions,
        "n_eval": len(rows),
        "layer0_metrics": metrics(y_true, layer0),
        f"layer{args.compare_layer}_metrics": metrics(y_true, compare),
        "layer0_error_count": len(errors),
        "correlations_with_layer0_p_harm": {
            "tfidf_p_harm_pearson": pearson(layer0_for_baselines, tfidf_scores),
            "tfidf_p_harm_spearman": spearman(layer0_for_baselines, tfidf_scores),
            "length_logreg_p_harm_pearson": pearson(layer0_for_baselines, length_scores),
            "length_logreg_p_harm_spearman": spearman(layer0_for_baselines, length_scores),
            "char_len_pearson": pearson(layer0, char_lens),
            "char_len_spearman": spearman(layer0, char_lens),
            "qwen_token_count_pearson": pearson(layer0_for_tokens, token_counts),
            "qwen_token_count_spearman": spearman(layer0_for_tokens, token_counts),
        },
        "files": {
            "all_scores": str(out_dir / "layer0_eval_scores.csv"),
            "top_harm_like": str(out_dir / "layer0_top_harm_like.csv"),
            "top_safe_like": str(out_dir / "layer0_top_safe_like.csv"),
            "confident_errors": str(out_dir / "layer0_confident_errors.csv"),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report = [
        "# Layer-0 Diagnostics",
        "",
        f"- Eval records: `{len(rows)}`",
        f"- Layer 0 AUROC/accuracy/F1: `{summary['layer0_metrics']['auroc']:.4f}` / `{summary['layer0_metrics']['accuracy']:.4f}` / `{summary['layer0_metrics']['f1']:.4f}`",
        f"- Layer {args.compare_layer} AUROC/accuracy/F1: `{summary[f'layer{args.compare_layer}_metrics']['auroc']:.4f}` / `{summary[f'layer{args.compare_layer}_metrics']['accuracy']:.4f}` / `{summary[f'layer{args.compare_layer}_metrics']['f1']:.4f}`",
        f"- Layer 0 errors: `{len(errors)}`",
        "",
        "## Correlations With Layer-0 Harm Score",
        "",
    ]
    for key, value in summary["correlations_with_layer0_p_harm"].items():
        rendered = "null" if value is None or (isinstance(value, float) and math.isnan(value)) else f"{value:.4f}"
        report.append(f"- `{key}`: `{rendered}`")
    report.extend(
        [
            "",
            "See the CSV files for manual inspection of top harm-like, top safe-like, and confident layer-0 errors.",
            "",
        ]
    )
    (out_dir / "report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
