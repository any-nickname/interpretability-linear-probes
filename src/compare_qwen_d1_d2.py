"""
Compare current Qwen D1 and D2 probes on shared data.

D1 predicts prompt label: harm vs safe.
D2 predicts response behavior: refusal vs compliance.

This script is analysis/reporting only. It does not retrain probes.
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
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def csv_float(row: dict, *keys: str) -> float:
    for key in keys:
        if key in row and row[key] not in ("", None):
            return float(row[key])
    raise KeyError(keys[0])


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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
    pearson = float(np.corrcoef(x, y)[0, 1])
    spearman = float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])
    return {"pearson": pearson, "spearman": spearman}


def binary_label_metrics(rows: list[dict], name: str) -> dict:
    usable = [
        row for row in rows if row.get("behavior_label") in ("refusal", "compliance")
    ]
    y_prompt = np.array([1 if row["prompt_label"] == "harm" else 0 for row in usable], dtype=int)
    y_behavior = np.array(
        [1 if row["behavior_label"] == "refusal" else 0 for row in usable], dtype=int
    )
    counts = Counter((row["prompt_label"], row["behavior_label"]) for row in usable)
    by_prompt = {
        label: dict(Counter(row["behavior_label"] for row in usable if row["prompt_label"] == label))
        for label in ("harm", "safe")
    }
    return {
        "dataset": name,
        "n_binary": int(len(usable)),
        "prompt_behavior_counts": {"|".join(k): v for k, v in sorted(counts.items())},
        "behavior_by_prompt_label": by_prompt,
        "accuracy_if_refusal_equals_harm": float(accuracy_score(y_prompt, y_behavior)),
        "mcc_prompt_label_vs_behavior": float(matthews_corrcoef(y_prompt, y_behavior)),
        "auroc_prompt_label_predicts_behavior": float(roc_auc_score(y_behavior, y_prompt))
        if len(set(y_behavior.tolist())) >= 2
        else math.nan,
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


def probe_proba(layer_payload: dict, acts: torch.Tensor, layer: int) -> np.ndarray:
    x = acts[:, layer, :].to(torch.float32).numpy()
    mean = layer_payload["scaler_mean"].to(torch.float32).numpy()
    scale = layer_payload["scaler_scale"].to(torch.float32).numpy()
    coef = layer_payload["coef"].to(torch.float32).numpy()
    intercept = float(layer_payload["intercept"])
    return sigmoid(((x - mean) / scale) @ coef + intercept)


def raw_direction(layer_payload: dict) -> np.ndarray:
    coef = layer_payload["coef"].to(torch.float32).numpy()
    scale = layer_payload["scaler_scale"].to(torch.float32).numpy()
    return coef / np.where(scale == 0.0, 1.0, scale)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return math.nan
    return float(np.dot(a, b) / denom)


def load_shared_rows(path: Path) -> list[dict]:
    rows = [
        row for row in read_jsonl(path) if row["behavior_label"] in ("refusal", "compliance")
    ]
    rows.sort(key=lambda row: row["id"])
    return rows


def summarize_error_overlap(rows: list[dict], split: str) -> dict:
    sub = [row for row in rows if row["split"] == split]
    return {
        "split": split,
        "n": len(sub),
        "counts": {
            "|".join(map(str, key)): value
            for key, value in sorted(
                Counter(
                    (
                        row["prompt_label"],
                        row["behavior_label"],
                        row["d1_correct"],
                        row["d2_correct"],
                    )
                    for row in sub
                ).items()
            )
        },
        "d1_error_count": sum(1 for row in sub if not row["d1_correct"]),
        "d2_error_count": sum(1 for row in sub if not row["d2_correct"]),
        "both_error_count": sum(
            1 for row in sub if not row["d1_correct"] and not row["d2_correct"]
        ),
        "either_error_count": sum(
            1 for row in sub if (not row["d1_correct"]) or (not row["d2_correct"])
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--d1-probe", default="data/probes/itw_manual_d1/qwen.pt")
    parser.add_argument("--d2-probe", default="data/probes/itw_manual_d2/qwen.pt")
    parser.add_argument("--d1-selected-layer", type=int, default=25)
    parser.add_argument("--d2-selected-layer", type=int, default=18)
    parser.add_argument("--d2-itw-control-layer", type=int, default=20)
    parser.add_argument(
        "--d2-ood-all",
        default="data/responses/d2_ood_balanced_200/qwen/qwen_d2_ood_balanced_200_all_labeled.jsonl",
    )
    parser.add_argument(
        "--itw-d2-all",
        default="data/responses/itw_manual_d2/qwen/itw_manual_d2_all_labeled.jsonl",
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
    parser.add_argument("--out-dir", default="data/analysis/qwen_d1_d2_comparison")
    args = parser.parse_args()

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    d1 = torch.load(ROOT / args.d1_probe, weights_only=False)
    d2 = torch.load(ROOT / args.d2_probe, weights_only=False)
    n_layers = min(int(d1["n_layers"]), int(d2["n_layers"]))

    # Same-layer geometry.
    cosine_rows = []
    for layer in range(n_layers):
        d1_dir = raw_direction(d1["layers"][layer])
        d2_dir = raw_direction(d2["layers"][layer])
        cosine_rows.append(
            {
                "layer": layer,
                "cosine_raw_space": cosine(d1_dir, d2_dir),
                "abs_cosine_raw_space": abs(cosine(d1_dir, d2_dir)),
            }
        )
    write_csv(out_dir / "same_layer_cosine.csv", cosine_rows, list(cosine_rows[0].keys()))

    # Prompt-label vs response-behavior association.
    itw_behavior = binary_label_metrics(read_jsonl(ROOT / args.itw_d2_all), "itw_core")
    ood_behavior = binary_label_metrics(read_jsonl(ROOT / args.d2_ood_all), "ood_balanced_200")

    # Shared OOD set score comparison.
    shared_rows = load_shared_rows(ROOT / args.d2_ood_all)
    activation_index = load_activation_index([ROOT / p for p in args.activation_files])
    missing = [row["id"] for row in shared_rows if row["id"] not in activation_index]
    if missing:
        raise ValueError(f"missing activations: {missing[:10]}")
    acts = torch.stack([activation_index[row["id"]] for row in shared_rows]).to(torch.float32)

    shared_table = []
    d1_selected = probe_proba(d1["layers"][args.d1_selected_layer], acts, args.d1_selected_layer)
    d2_selected = probe_proba(d2["layers"][args.d2_selected_layer], acts, args.d2_selected_layer)
    d2_control = probe_proba(d2["layers"][args.d2_itw_control_layer], acts, args.d2_itw_control_layer)
    y_prompt = np.array([1 if row["prompt_label"] == "harm" else 0 for row in shared_rows], dtype=int)
    y_behavior = np.array(
        [1 if row["behavior_label"] == "refusal" else 0 for row in shared_rows], dtype=int
    )
    splits = [row["d2_ood_split"] for row in shared_rows]
    for idx, row in enumerate(shared_rows):
        d1_pred = int(d1_selected[idx] >= 0.5)
        d2_pred = int(d2_selected[idx] >= 0.5)
        shared_table.append(
            {
                "id": row["id"],
                "split": splits[idx],
                "source": row.get("source"),
                "prompt_label": row["prompt_label"],
                "behavior_label": row["behavior_label"],
                "manual_label_source": row.get("manual_label_source"),
                "d1_layer": args.d1_selected_layer,
                "d1_p_harm": float(d1_selected[idx]),
                "d1_pred_prompt_label": "harm" if d1_pred else "safe",
                "d1_correct": bool(d1_pred == y_prompt[idx]),
                "d2_layer": args.d2_selected_layer,
                "d2_p_refusal": float(d2_selected[idx]),
                "d2_pred_behavior": "refusal" if d2_pred else "compliance",
                "d2_correct": bool(d2_pred == y_behavior[idx]),
                f"d2_layer{args.d2_itw_control_layer}_p_refusal": float(d2_control[idx]),
            }
        )
    write_csv(out_dir / "shared_ood_scores.csv", shared_table, list(shared_table[0].keys()))

    score_corr_rows = []
    for split in ("ood_dev", "ood_test", "all"):
        idxs = (
            [idx for idx, row in enumerate(shared_table) if row["split"] == split]
            if split != "all"
            else list(range(len(shared_table)))
        )
        corr = corr_pair(d1_selected[idxs], d2_selected[idxs])
        score_corr_rows.append(
            {
                "split": split,
                "n": len(idxs),
                "d1_layer": args.d1_selected_layer,
                "d2_layer": args.d2_selected_layer,
                **corr,
            }
        )
    for layer in range(n_layers):
        d1_scores = probe_proba(d1["layers"][layer], acts, layer)
        d2_scores = probe_proba(d2["layers"][layer], acts, layer)
        corr = corr_pair(d1_scores, d2_scores)
        score_corr_rows.append(
            {
                "split": "all",
                "n": len(shared_rows),
                "d1_layer": layer,
                "d2_layer": layer,
                **corr,
            }
        )
    write_csv(out_dir / "score_correlations.csv", score_corr_rows, list(score_corr_rows[0].keys()))

    error_overlap = [
        summarize_error_overlap(shared_table, "ood_dev"),
        summarize_error_overlap(shared_table, "ood_test"),
    ]

    # Layer AUROC tables pulled from current saved results.
    d1_itw_layers = read_csv(ROOT / "data/probes/itw_manual_d1/qwen_layers.csv")
    d2_itw_layers = read_csv(ROOT / "data/probes/itw_manual_d2/qwen_layers.csv")
    d1_ood_layers = read_csv(
        ROOT
        / "data/validation/itw_manual_d1/qwen/ood_clean/ood_dev_test_layer_selection/layer_selection_summary.csv"
    )
    d2_ood_layers = read_csv(
        ROOT
        / "data/validation/itw_manual_d2/qwen/ood_balanced_200/layer_selection/qwen_d2_ood_balanced_200_layers.csv"
    )

    layer_summary_rows = []
    for layer in range(n_layers):
        d1_itw = next(row for row in d1_itw_layers if int(row["layer"]) == layer)
        d2_itw = next(row for row in d2_itw_layers if int(row["layer"]) == layer)
        d1_ood = next(row for row in d1_ood_layers if int(row["layer"]) == layer)
        d2_ood = next(row for row in d2_ood_layers if int(row["layer"]) == layer)
        layer_summary_rows.append(
            {
                "layer": layer,
                "d1_itw_eval_auroc": csv_float(d1_itw, "eval_auroc", "auroc"),
                "d1_ood_dev_mean_auroc": float(d1_ood["mean_dev_auroc"]),
                "d1_ood_test_mean_auroc": float(d1_ood["mean_test_auroc"]),
                "d2_itw_eval_auroc": csv_float(d2_itw, "eval_auroc", "auroc"),
                "d2_ood_dev_auroc": float(d2_ood["dev_auroc"]),
                "d2_ood_test_auroc": float(d2_ood["test_auroc"]),
                "same_layer_cosine_raw_space": float(cosine_rows[layer]["cosine_raw_space"]),
            }
        )
    write_csv(out_dir / "layerwise_summary.csv", layer_summary_rows, list(layer_summary_rows[0].keys()))

    selected_cosine = next(
        row for row in cosine_rows if int(row["layer"]) == args.d2_selected_layer
    )["cosine_raw_space"]
    d1_selected_layer_cosine = next(
        row for row in cosine_rows if int(row["layer"]) == args.d1_selected_layer
    )["cosine_raw_space"]
    selected_score_corr_test = next(
        row for row in score_corr_rows if row["split"] == "ood_test" and row["d1_layer"] == args.d1_selected_layer
    )

    summary = {
        "d1_selected_layer": args.d1_selected_layer,
        "d2_selected_layer": args.d2_selected_layer,
        "d2_itw_control_layer": args.d2_itw_control_layer,
        "prompt_behavior_association": {
            "itw_core": itw_behavior,
            "ood_balanced_200": ood_behavior,
        },
        "selected_score_correlation_ood_test": selected_score_corr_test,
        "same_layer_cosine_at_d2_selected_layer_18": selected_cosine,
        "same_layer_cosine_at_d1_selected_layer_25": d1_selected_layer_cosine,
        "error_overlap": error_overlap,
        "notes": [
            "Cosine is computed only within the same layer and uses raw-space directions coef/scaler_scale.",
            "D1 selected layer 25 and D2 selected layer 18 are not directly compared by cosine because they live in different layers.",
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    def fmt(value: float) -> str:
        return "nan" if value != value else f"{value:.4f}"

    report = [
        "# Qwen D1 vs D2 Comparison",
        "",
        "## Current selected probes",
        "",
        f"- D1 prompt-label probe: OOD-dev-selected layer `{args.d1_selected_layer}`",
        f"- D2 response-behavior probe: OOD-dev-selected layer `{args.d2_selected_layer}`",
        f"- D2 ITW-eval-selected control layer: `{args.d2_itw_control_layer}`",
        "",
        "## Prompt label vs response behavior",
        "",
        "| Dataset | n | accuracy if refusal=harm | MCC | AUROC prompt-label->behavior |",
        "|---|---:|---:|---:|---:|",
        f"| ITW-core | {itw_behavior['n_binary']} | {itw_behavior['accuracy_if_refusal_equals_harm']:.4f} | {itw_behavior['mcc_prompt_label_vs_behavior']:.4f} | {itw_behavior['auroc_prompt_label_predicts_behavior']:.4f} |",
        f"| OOD-balanced-200 | {ood_behavior['n_binary']} | {ood_behavior['accuracy_if_refusal_equals_harm']:.4f} | {ood_behavior['mcc_prompt_label_vs_behavior']:.4f} | {ood_behavior['auroc_prompt_label_predicts_behavior']:.4f} |",
        "",
        "## Score correlation on shared OOD-balanced prompts",
        "",
        f"- Selected D1 layer `{args.d1_selected_layer}` harm-score vs selected D2 layer `{args.d2_selected_layer}` refusal-score on OOD-test: Pearson `{fmt(selected_score_corr_test['pearson'])}`, Spearman `{fmt(selected_score_corr_test['spearman'])}`.",
        "",
        "## Same-layer cosine",
        "",
        f"- Same-layer cosine at D2-selected layer `18`: `{selected_cosine:.4f}`.",
        f"- Same-layer cosine at D1-selected layer `25`: `{d1_selected_layer_cosine:.4f}`.",
        "- Cross-layer cosine between D1 layer 25 and D2 layer 18 is intentionally not reported.",
        "",
        "## Error overlap on shared OOD-balanced prompts",
        "",
        "| Split | n | D1 errors | D2 errors | Both errors | Either errors |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in error_overlap:
        report.append(
            f"| {row['split']} | {row['n']} | {row['d1_error_count']} | {row['d2_error_count']} | {row['both_error_count']} | {row['either_error_count']} |"
        )
    report.append("")
    report.append("Full CSV outputs are saved next to this report.")
    report.append("")
    (out_dir / "report.md").write_text("\n".join(report), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"wrote {(out_dir / 'report.md').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
