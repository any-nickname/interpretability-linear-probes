"""
Train exact ITW D1 text baselines on the train split and evaluate them on clean
OOD datasets.

This mirrors the exact-split text baseline, but the evaluation files are the
clean OOD JSONL files materialized from raw sources.
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


def labels_to_y(rows: list[dict]) -> list[int]:
    return [1 if row["label"] == "harm" else 0 for row in rows]


def try_load_tokenizer():
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen3-1.7B",
            trust_remote_code=True,
            local_files_only=True,
        )
        return tokenizer, "Qwen/Qwen3-1.7B chat template"
    except Exception as exc:
        return None, f"fallback_whitespace_token_count: {type(exc).__name__}: {exc}"


def qwen_token_count(tokenizer, prompt: str) -> int:
    if tokenizer is None:
        return len(prompt.split())
    messages = [{"role": "user", "content": prompt}]
    token_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
    )
    if hasattr(token_ids, "input_ids"):
        token_ids = token_ids.input_ids
    elif isinstance(token_ids, dict):
        token_ids = token_ids["input_ids"]
    if token_ids and isinstance(token_ids[0], list):
        token_ids = token_ids[0]
    return len(token_ids)


def metrics_from_scores(y_true, scores, threshold=0.5) -> dict:
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

    preds = [1 if score >= threshold else 0 for score in scores]
    return {
        "auroc": float(roc_auc_score(y_true, scores)),
        "accuracy": float(accuracy_score(y_true, preds)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
    }


def dataset_name(path: Path) -> str:
    return path.stem.removesuffix("_clean")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="datasets/processed/itw_manual_d1_train.jsonl")
    parser.add_argument("--ood-dir", default="datasets/processed/ood_clean")
    parser.add_argument("--out-dir", default="data/validation/itw_manual_d1/qwen/ood_clean/text_baselines")
    args = parser.parse_args()

    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    train_rows = read_jsonl(ROOT / args.train)
    y_train = labels_to_y(train_rows)
    train_texts = [row["prompt"] for row in train_rows]
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    vectorizer = TfidfVectorizer(ngram_range=(1, 5), lowercase=True, sublinear_tf=True, norm="l2")
    x_train = vectorizer.fit_transform(train_texts)
    tfidf_model = LogisticRegression(solver="liblinear", max_iter=1000, random_state=0)
    tfidf_model.fit(x_train, y_train)

    tokenizer, token_count_source = try_load_tokenizer()

    def feature_row(row: dict) -> list[float]:
        prompt = row["prompt"]
        return [
            float(len(prompt)),
            float(len(prompt.split())),
            float(qwen_token_count(tokenizer, prompt)),
        ]

    length_train = np.array([feature_row(row) for row in train_rows], dtype=float)
    length_model = make_pipeline(
        StandardScaler(),
        LogisticRegression(solver="liblinear", max_iter=1000, random_state=0),
    )
    length_model.fit(length_train, y_train)

    summaries = []
    ood_paths = sorted((ROOT / args.ood_dir).glob("*.jsonl"))
    if not ood_paths:
        raise FileNotFoundError(ROOT / args.ood_dir)

    for path in ood_paths:
        rows = read_jsonl(path)
        y_eval = labels_to_y(rows)
        texts = [row["prompt"] for row in rows]
        x_eval = vectorizer.transform(texts)
        tfidf_scores = tfidf_model.predict_proba(x_eval)[:, 1]

        length_eval = np.array([feature_row(row) for row in rows], dtype=float)
        length_scores = length_model.predict_proba(length_eval)[:, 1]

        summary = {
            "dataset": dataset_name(path),
            "path": str(path.relative_to(ROOT)),
            "n_records": len(rows),
            "n_harm": int(sum(y_eval)),
            "n_safe": int(len(y_eval) - sum(y_eval)),
            "tfidf_logreg": metrics_from_scores(y_eval, tfidf_scores),
            "length_token_logreg": metrics_from_scores(y_eval, length_scores),
            "raw_length_scores": {
                "char_len_as_harm_score": metrics_from_scores(y_eval, length_eval[:, 0]),
                "word_len_as_harm_score": metrics_from_scores(y_eval, length_eval[:, 1]),
                "qwen_token_count_as_harm_score": metrics_from_scores(y_eval, length_eval[:, 2]),
            },
        }
        summaries.append(summary)

        prediction_rows = []
        for idx, row in enumerate(rows):
            prediction_rows.append(
                {
                    "id": row["id"],
                    "source": row.get("source", ""),
                    "label": row["label"],
                    "y": y_eval[idx],
                    "tfidf_p_harm": float(tfidf_scores[idx]),
                    "length_logreg_p_harm": float(length_scores[idx]),
                    "char_len": int(length_eval[idx, 0]),
                    "word_len": int(length_eval[idx, 1]),
                    "qwen_token_count": int(length_eval[idx, 2]),
                    "prompt_preview": " ".join(row["prompt"].split())[:300],
                }
            )
        write_csv(
            out_dir / summary["dataset"] / "predictions.csv",
            prediction_rows,
            [
                "id",
                "source",
                "label",
                "y",
                "tfidf_p_harm",
                "length_logreg_p_harm",
                "char_len",
                "word_len",
                "qwen_token_count",
                "prompt_preview",
            ],
        )
        (out_dir / summary["dataset"] / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(
            f"{summary['dataset']}: TF-IDF AUROC={summary['tfidf_logreg']['auroc']:.4f}; "
            f"length/token AUROC={summary['length_token_logreg']['auroc']:.4f}",
            flush=True,
        )

    write_csv(
        out_dir / "ood_text_baseline_summary.csv",
        [
            {
                "dataset": s["dataset"],
                "n_records": s["n_records"],
                "n_harm": s["n_harm"],
                "n_safe": s["n_safe"],
                "tfidf_auroc": s["tfidf_logreg"]["auroc"],
                "tfidf_accuracy": s["tfidf_logreg"]["accuracy"],
                "tfidf_f1": s["tfidf_logreg"]["f1"],
                "length_token_auroc": s["length_token_logreg"]["auroc"],
                "char_len_auroc": s["raw_length_scores"]["char_len_as_harm_score"]["auroc"],
                "word_len_auroc": s["raw_length_scores"]["word_len_as_harm_score"]["auroc"],
                "qwen_token_count_auroc": s["raw_length_scores"]["qwen_token_count_as_harm_score"]["auroc"],
            }
            for s in summaries
        ],
        [
            "dataset",
            "n_records",
            "n_harm",
            "n_safe",
            "tfidf_auroc",
            "tfidf_accuracy",
            "tfidf_f1",
            "length_token_auroc",
            "char_len_auroc",
            "word_len_auroc",
            "qwen_token_count_auroc",
        ],
    )

    payload = {
        "train": args.train,
        "ood_dir": args.ood_dir,
        "out_dir": args.out_dir,
        "token_count_source": token_count_source,
        "datasets": summaries,
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    report = [
        "# Clean OOD Text Baselines",
        "",
        f"- Train split: `{args.train}`",
        f"- OOD source: `{args.ood_dir}`",
        f"- Token count source: `{token_count_source}`",
        "",
        "| Dataset | n | harm | safe | TF-IDF AUROC | length/token AUROC | Qwen-token AUROC |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for s in summaries:
        report.append(
            f"| `{s['dataset']}` | {s['n_records']} | {s['n_harm']} | {s['n_safe']} | "
            f"{s['tfidf_logreg']['auroc']:.4f} | {s['length_token_logreg']['auroc']:.4f} | "
            f"{s['raw_length_scores']['qwen_token_count_as_harm_score']['auroc']:.4f} |"
        )
    report.append("")
    (out_dir / "report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
