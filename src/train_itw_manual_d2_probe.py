"""
Train a layer-wise Qwen D2 response-behavior probe on manually labeled ITW data.

D2 predicts model response behavior (`refusal` vs `compliance`) from prompt
activations. `partial` and `unclear` labels are excluded from the first binary
D2-core run.
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_labels(path: Path) -> dict[str, dict]:
    return {row["id"]: row for row in read_jsonl(path)}


def usable_rows(act_data: dict, labels: dict[str, dict]) -> list[dict]:
    rows = []
    for idx, rec_id in enumerate(act_data["ids"]):
        label_row = labels.get(rec_id)
        if label_row is None:
            continue
        behavior = label_row["behavior_label"]
        if behavior not in ("refusal", "compliance"):
            continue
        rows.append(
            {
                "idx": idx,
                "id": rec_id,
                "behavior_label": behavior,
                "prompt_label": label_row.get("prompt_label"),
                "pair_id": label_row.get("pair_id"),
            }
        )
    return rows


def labels_to_y(rows: list[dict]) -> np.ndarray:
    return np.array([1 if row["behavior_label"] == "refusal" else 0 for row in rows], dtype=int)


def slice_layer(act_data: dict, rows: list[dict], layer: int) -> np.ndarray:
    acts = act_data["activations"]
    return np.array(
        [acts[row["idx"], layer, :].to(torch.float32).numpy() for row in rows],
        dtype=np.float32,
    )


def evaluate(probe, scaler, x_eval: np.ndarray, y_eval: np.ndarray) -> dict:
    xs_eval = scaler.transform(x_eval)
    proba = probe.predict_proba(xs_eval)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "auroc": float(roc_auc_score(y_eval, proba)),
        "accuracy": float(accuracy_score(y_eval, pred)),
        "f1": float(f1_score(y_eval, pred, zero_division=0)),
        "precision": float(precision_score(y_eval, pred, zero_division=0)),
        "recall": float(recall_score(y_eval, pred, zero_division=0)),
        "mean_p_refusal": float(proba.mean()),
    }


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen")
    parser.add_argument("--train-activations", default="data/activations/qwen/itw_manual_d1_train.pt")
    parser.add_argument("--eval-activations", default="data/activations/qwen/itw_manual_d1_eval.pt")
    parser.add_argument("--train-labels", default="data/responses/itw_manual_d2/qwen/itw_manual_d2_train_labeled.jsonl")
    parser.add_argument("--eval-labels", default="data/responses/itw_manual_d2/qwen/itw_manual_d2_eval_labeled.jsonl")
    parser.add_argument("--out-dir", default="data/probes/itw_manual_d2")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    train_acts = torch.load(ROOT / args.train_activations, weights_only=False)
    eval_acts = torch.load(ROOT / args.eval_activations, weights_only=False)
    train_labels = load_labels(ROOT / args.train_labels)
    eval_labels = load_labels(ROOT / args.eval_labels)

    train_rows = usable_rows(train_acts, train_labels)
    eval_rows = usable_rows(eval_acts, eval_labels)
    y_train = labels_to_y(train_rows)
    y_eval = labels_to_y(eval_rows)

    if len(set(y_train.tolist())) < 2:
        raise ValueError("D2 train split has fewer than two behavior classes")
    if len(set(y_eval.tolist())) < 2:
        raise ValueError("D2 eval split has fewer than two behavior classes")

    n_layers = int(train_acts["n_layers"])
    d_model = int(train_acts["d_model"])

    rows = []
    layer_payload = []
    prediction_rows = []
    for layer in range(n_layers):
        x_train = slice_layer(train_acts, train_rows, layer)
        x_eval = slice_layer(eval_acts, eval_rows, layer)

        scaler = StandardScaler()
        xs_train = scaler.fit_transform(x_train)
        probe = LogisticRegression(
            random_state=args.seed,
            max_iter=2000,
            C=1.0,
            solver="lbfgs",
            class_weight="balanced",
        )
        probe.fit(xs_train, y_train)

        train_metrics = evaluate(probe, scaler, x_train, y_train)
        eval_metrics = evaluate(probe, scaler, x_eval, y_eval)
        row = {
            "model": args.model,
            "layer": layer,
            "n_train": int(len(y_train)),
            "n_eval": int(len(y_eval)),
            "train_refusal": int(y_train.sum()),
            "eval_refusal": int(y_eval.sum()),
            "train_compliance": int(len(y_train) - y_train.sum()),
            "eval_compliance": int(len(y_eval) - y_eval.sum()),
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"eval_{k}": v for k, v in eval_metrics.items()},
        }
        rows.append(row)
        layer_payload.append(
            {
                "layer": layer,
                "coef": torch.tensor(probe.coef_[0], dtype=torch.float32),
                "intercept": float(probe.intercept_[0]),
                "scaler_mean": torch.tensor(scaler.mean_, dtype=torch.float32),
                "scaler_scale": torch.tensor(scaler.scale_, dtype=torch.float32),
                "metrics_train": train_metrics,
                "metrics_eval": eval_metrics,
            }
        )
        print(
            f"layer {layer:>2}: eval AUROC={eval_metrics['auroc']:.4f} "
            f"acc={eval_metrics['accuracy']:.4f} f1={eval_metrics['f1']:.4f}",
            flush=True,
        )

        if layer in (0,):
            proba = probe.predict_proba(scaler.transform(x_eval))[:, 1]
            for idx, eval_row in enumerate(eval_rows):
                prediction_rows.append(
                    {
                        "id": eval_row["id"],
                        "prompt_label": eval_row["prompt_label"],
                        "behavior_label": eval_row["behavior_label"],
                        "y": int(y_eval[idx]),
                        "layer0_p_refusal": float(proba[idx]),
                    }
                )

    csv_path = out_dir / f"{args.model}_layers.csv"
    write_csv(csv_path, rows, list(rows[0].keys()))

    best = max(rows, key=lambda row: row["eval_auroc"])
    summary = {
        "model": args.model,
        "probe_type": "D2 response behavior",
        "positive_class": "refusal",
        "negative_class": "compliance",
        "excluded_labels": ["partial", "unclear"],
        "class_weight": "balanced",
        "n_layers": n_layers,
        "d_model": d_model,
        "n_train_usable": int(len(y_train)),
        "n_eval_usable": int(len(y_eval)),
        "train_refusal": int(y_train.sum()),
        "eval_refusal": int(y_eval.sum()),
        "train_compliance": int(len(y_train) - y_train.sum()),
        "eval_compliance": int(len(y_eval) - y_eval.sum()),
        "best_layer": int(best["layer"]),
        "best_eval_auroc": float(best["eval_auroc"]),
        "best_eval_accuracy": float(best["eval_accuracy"]),
        "best_eval_f1": float(best["eval_f1"]),
    }
    json_path = out_dir / f"{args.model}_summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    pt_path = out_dir / f"{args.model}.pt"
    torch.save(
        {
            "model": args.model,
            "train_activations": args.train_activations,
            "eval_activations": args.eval_activations,
            "train_labels": args.train_labels,
            "eval_labels": args.eval_labels,
            "n_layers": n_layers,
            "d_model": d_model,
            "layers": layer_payload,
            "summary": summary,
        },
        pt_path,
    )

    report = [
        "# Qwen ITW Manual D2 Probe",
        "",
        "- Positive class: `refusal`",
        "- Negative class: `compliance`",
        "- Excluded labels: `partial`, `unclear`",
        "- LogisticRegression class weight: `balanced`",
        f"- Train usable records: `{len(y_train)}` (`{int(y_train.sum())}` refusal / `{int(len(y_train) - y_train.sum())}` compliance)",
        f"- Eval usable records: `{len(y_eval)}` (`{int(y_eval.sum())}` refusal / `{int(len(y_eval) - y_eval.sum())}` compliance)",
        f"- Best eval layer: `{summary['best_layer']}`",
        f"- Best eval AUROC/accuracy/F1: `{summary['best_eval_auroc']:.4f}` / `{summary['best_eval_accuracy']:.4f}` / `{summary['best_eval_f1']:.4f}`",
        "",
    ]
    (out_dir / f"{args.model}_report.md").write_text("\n".join(report), encoding="utf-8")

    print(f"best layer={best['layer']} eval AUROC={best['eval_auroc']:.4f}")
    print(f"wrote {csv_path.relative_to(ROOT)}")
    print(f"wrote {json_path.relative_to(ROOT)}")
    print(f"wrote {pt_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
