"""
Exact train/eval text baselines for the manual In-the-Wild D1 split.

This is different from the LOPO TF-IDF dataset audit: it uses the same
train/eval JSONL files as the activation probe, so the numbers are directly
comparable to D1 probe eval metrics.
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
    except Exception as exc:  # pragma: no cover - diagnostic fallback
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="datasets/processed/itw_manual_d1_train.jsonl")
    parser.add_argument("--eval", default="datasets/processed/itw_manual_d1_eval.jsonl")
    parser.add_argument("--out-dir", default="data/validation/itw_manual_d1/qwen/text_baselines")
    args = parser.parse_args()

    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    train_path = ROOT / args.train
    eval_path = ROOT / args.eval
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    train_rows = read_jsonl(train_path)
    eval_rows = read_jsonl(eval_path)
    y_train = labels_to_y(train_rows)
    y_eval = labels_to_y(eval_rows)
    train_texts = [row["prompt"] for row in train_rows]
    eval_texts = [row["prompt"] for row in eval_rows]

    vectorizer = TfidfVectorizer(ngram_range=(1, 5), lowercase=True, sublinear_tf=True, norm="l2")
    x_train = vectorizer.fit_transform(train_texts)
    x_eval = vectorizer.transform(eval_texts)
    tfidf_model = LogisticRegression(solver="liblinear", max_iter=1000, random_state=0)
    tfidf_model.fit(x_train, y_train)
    tfidf_scores = tfidf_model.predict_proba(x_eval)[:, 1]
    tfidf_metrics = metrics_from_scores(y_eval, tfidf_scores)
    tfidf_train_acc = float(accuracy_score(y_train, tfidf_model.predict(x_train)))

    tokenizer, token_count_source = try_load_tokenizer()

    def feature_row(row: dict) -> list[float]:
        prompt = row["prompt"]
        return [
            float(len(prompt)),
            float(len(prompt.split())),
            float(qwen_token_count(tokenizer, prompt)),
        ]

    length_train = np.array([feature_row(row) for row in train_rows], dtype=float)
    length_eval = np.array([feature_row(row) for row in eval_rows], dtype=float)

    length_model = make_pipeline(
        StandardScaler(),
        LogisticRegression(solver="liblinear", max_iter=1000, random_state=0),
    )
    length_model.fit(length_train, y_train)
    length_scores = length_model.predict_proba(length_eval)[:, 1]
    length_metrics = metrics_from_scores(y_eval, length_scores)
    length_train_acc = float(accuracy_score(y_train, length_model.predict(length_train)))

    char_metrics = metrics_from_scores(y_eval, length_eval[:, 0])
    word_metrics = metrics_from_scores(y_eval, length_eval[:, 1])
    token_metrics = metrics_from_scores(y_eval, length_eval[:, 2])

    majority_label = 1 if sum(y_train) >= len(y_train) / 2 else 0
    majority_preds = [majority_label for _ in y_eval]
    majority_accuracy = float(accuracy_score(y_eval, majority_preds))

    prediction_rows = []
    for idx, row in enumerate(eval_rows):
        prediction_rows.append(
            {
                "id": row["id"],
                "pair_id": row["pair_id"],
                "label": row["label"],
                "y": y_eval[idx],
                "tfidf_p_harm": float(tfidf_scores[idx]),
                "length_logreg_p_harm": float(length_scores[idx]),
                "char_len": int(length_eval[idx, 0]),
                "word_len": int(length_eval[idx, 1]),
                "qwen_token_count": int(length_eval[idx, 2]),
                "prompt_preview": " ".join(row["prompt"].split())[:260],
            }
        )
    write_csv(
        out_dir / "eval_predictions.csv",
        prediction_rows,
        [
            "id",
            "pair_id",
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

    feature_names = vectorizer.get_feature_names_out()
    coefs = tfidf_model.coef_[0]
    top_rows = []
    for i in np.argsort(coefs)[::-1][:100]:
        top_rows.append({"side": "harm", "feature": feature_names[i], "coef": float(coefs[i])})
    for i in np.argsort(coefs)[:100]:
        top_rows.append({"side": "safe", "feature": feature_names[i], "coef": float(coefs[i])})
    write_csv(out_dir / "tfidf_top_features.csv", top_rows, ["side", "feature", "coef"])

    metrics = {
        "train_path": args.train,
        "eval_path": args.eval,
        "out_dir": args.out_dir,
        "n_train": len(train_rows),
        "n_eval": len(eval_rows),
        "train_pairs": len(train_rows) // 2,
        "eval_pairs": len(eval_rows) // 2,
        "train_pos": int(sum(y_train)),
        "eval_pos": int(sum(y_eval)),
        "token_count_source": token_count_source,
        "tfidf_logreg": {
            "vectorizer": {"ngram_range": [1, 5], "lowercase": True, "sublinear_tf": True, "norm": "l2"},
            "train_accuracy": tfidf_train_acc,
            **tfidf_metrics,
        },
        "length_token_logreg": {
            "features": ["char_len", "word_len", "qwen_token_count"],
            "train_accuracy": length_train_acc,
            **length_metrics,
        },
        "raw_length_scores": {
            "char_len_as_harm_score": char_metrics,
            "word_len_as_harm_score": word_metrics,
            "qwen_token_count_as_harm_score": token_metrics,
        },
        "majority_baseline": {
            "majority_label": "harm" if majority_label == 1 else "safe",
            "accuracy": majority_accuracy,
        },
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    report = [
        "# Exact Train/Eval Text Baselines",
        "",
        f"- Train records: `{len(train_rows)}` (`{len(train_rows)//2}` pairs)",
        f"- Eval records: `{len(eval_rows)}` (`{len(eval_rows)//2}` pairs)",
        f"- Token count source: `{token_count_source}`",
        "",
        "## Metrics",
        "",
        f"- TF-IDF LogisticRegression AUROC: `{tfidf_metrics['auroc']:.4f}`",
        f"- TF-IDF LogisticRegression accuracy/F1: `{tfidf_metrics['accuracy']:.4f}` / `{tfidf_metrics['f1']:.4f}`",
        f"- Length/token LogisticRegression AUROC: `{length_metrics['auroc']:.4f}`",
        f"- Length/token LogisticRegression accuracy/F1: `{length_metrics['accuracy']:.4f}` / `{length_metrics['f1']:.4f}`",
        f"- Raw char length AUROC (longer = harm): `{char_metrics['auroc']:.4f}`",
        f"- Raw word length AUROC (longer = harm): `{word_metrics['auroc']:.4f}`",
        f"- Raw Qwen token count AUROC (more tokens = harm): `{token_metrics['auroc']:.4f}`",
        f"- Majority accuracy: `{majority_accuracy:.4f}`",
        "",
    ]
    (out_dir / "report.md").write_text("\n".join(report), encoding="utf-8")

    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
