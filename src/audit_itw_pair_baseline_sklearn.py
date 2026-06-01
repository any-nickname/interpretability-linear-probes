import argparse
import csv
import json
import statistics
import sys
from pathlib import Path


def add_workspace_deps(root):
    deps_path = root / ".deps" / "python"
    if deps_path.exists():
        sys.path.insert(0, str(deps_path))


def read_jsonl(path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_csv(path, rows, fieldnames):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sparse_centroid(matrix):
    centroid = matrix.mean(axis=0)
    from scipy import sparse
    from sklearn.preprocessing import normalize

    return normalize(sparse.csr_matrix(centroid))


def main():
    root = Path.cwd()
    add_workspace_deps(root)

    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, roc_auc_score

    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="datasets/processed/in_the_wild_source_prompts.jsonl")
    parser.add_argument("--pair-files", nargs="+", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--note", default="")
    args = parser.parse_args()

    source_path = root / args.source
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    sources = {row["id"]: row for row in read_jsonl(source_path)}
    pair_rows = []
    for pair_file in args.pair_files:
        pair_path = root / pair_file
        for row in read_jsonl(pair_path):
            row = dict(row)
            row["_pair_file"] = pair_file
            pair_rows.append(row)

    seen = set()
    duplicates = []
    pairs = []
    for row in pair_rows:
        harm_id = row["harm_id"]
        if harm_id in seen:
            duplicates.append(harm_id)
        seen.add(harm_id)
        if harm_id not in sources:
            raise SystemExit(f"missing source prompt for {harm_id}")
        pairs.append(
            {
                "harm_id": harm_id,
                "harm_prompt": sources[harm_id]["prompt"],
                "safe_prompt": row["safe_prompt"],
                "pair_file": row["_pair_file"],
            }
        )
    if duplicates:
        raise SystemExit(f"duplicate harm_id(s): {duplicates}")

    records = []
    for pair in pairs:
        records.append({"pair_id": pair["harm_id"], "label": "harm", "y": 1, "text": pair["harm_prompt"]})
        records.append({"pair_id": pair["harm_id"], "label": "safe", "y": 0, "text": pair["safe_prompt"]})

    y_true, centroid_scores, logreg_scores = [], [], []
    for heldout in [pair["harm_id"] for pair in pairs]:
        train = [record for record in records if record["pair_id"] != heldout]
        test = [record for record in records if record["pair_id"] == heldout]

        vectorizer = TfidfVectorizer(ngram_range=(1, 5), lowercase=True, sublinear_tf=True, norm="l2")
        x_train = vectorizer.fit_transform([record["text"] for record in train])
        y_train = np.array([record["y"] for record in train])
        x_test = vectorizer.transform([record["text"] for record in test])

        harm_centroid = sparse_centroid(x_train[y_train == 1])
        safe_centroid = sparse_centroid(x_train[y_train == 0])
        centroid = (x_test @ harm_centroid.T).toarray().ravel() - (x_test @ safe_centroid.T).toarray().ravel()

        model = LogisticRegression(solver="liblinear", max_iter=1000, random_state=0)
        model.fit(x_train, y_train)
        logreg = model.predict_proba(x_test)[:, 1]

        for record, c_score, l_score in zip(test, centroid, logreg):
            y_true.append(record["y"])
            centroid_scores.append(float(c_score))
            logreg_scores.append(float(l_score))

    y_all = [record["y"] for record in records]
    centroid_auc = roc_auc_score(y_true, centroid_scores)
    logreg_auc = roc_auc_score(y_true, logreg_scores)
    char_auc = roc_auc_score(y_all, [len(record["text"]) for record in records])
    word_auc = roc_auc_score(y_all, [len(record["text"].split()) for record in records])

    per_pair = []
    exact = []
    for pair in pairs:
        harm = pair["harm_prompt"]
        safe = pair["safe_prompt"]
        row = {
            "pair_id": pair["harm_id"],
            "pair_file": pair["pair_file"],
            "harm_chars": len(harm),
            "safe_chars": len(safe),
            "delta_chars_safe_minus_harm": len(safe) - len(harm),
            "harm_words": len(harm.split()),
            "safe_words": len(safe.split()),
            "delta_words_safe_minus_harm": len(safe.split()) - len(harm.split()),
            "exact_match": harm == safe,
        }
        if row["exact_match"]:
            exact.append(pair["harm_id"])
        per_pair.append(row)

    vectorizer = TfidfVectorizer(ngram_range=(1, 5), lowercase=True, sublinear_tf=True, norm="l2")
    x_full = vectorizer.fit_transform([record["text"] for record in records])
    y_full = np.array(y_all)
    model = LogisticRegression(solver="liblinear", max_iter=1000, random_state=0)
    model.fit(x_full, y_full)
    train_acc = accuracy_score(y_full, model.predict(x_full))

    feature_names = vectorizer.get_feature_names_out()
    coefs = model.coef_[0]
    top_harm_idx = np.argsort(coefs)[::-1][:100]
    top_safe_idx = np.argsort(coefs)[:100]
    top_rows = []
    top_safe = []
    top_harm = []
    for i in top_harm_idx:
        item = (feature_names[i], float(coefs[i]))
        top_harm.append(item)
        top_rows.append({"side": "harm", "feature": item[0], "coef": item[1]})
    for i in top_safe_idx:
        item = (feature_names[i], float(coefs[i]))
        top_safe.append(item)
        top_rows.append({"side": "safe", "feature": item[0], "coef": item[1]})

    write_csv(
        out_dir / "pair_scores.csv",
        per_pair,
        [
            "pair_id",
            "pair_file",
            "harm_chars",
            "safe_chars",
            "delta_chars_safe_minus_harm",
            "harm_words",
            "safe_words",
            "delta_words_safe_minus_harm",
            "exact_match",
        ],
    )
    write_csv(out_dir / "tfidf_top_features.csv", top_rows, ["side", "feature", "coef"])

    metrics = {
        "source_path": args.source,
        "pair_files": args.pair_files,
        "n_pairs": len(pairs),
        "n_records": len(records),
        "included_ids": [pair["harm_id"] for pair in pairs],
        "note": args.note,
        "implementation_note": "Fast sklearn TF-IDF/logistic baseline; separate audit from the standard pure-Python LOPO script.",
        "dependencies": {
            "sklearn": __import__("sklearn").__version__,
            "numpy": np.__version__,
        },
        "tfidf": {
            "vectorizer": {"ngram_range": [1, 5], "lowercase": True, "sublinear_tf": True, "norm": "l2"},
            "centroid_lopo_auroc": float(centroid_auc),
            "logreg_lopo_auroc": float(logreg_auc),
            "logreg_full_train_accuracy": float(train_acc),
            "logreg": {"solver": "liblinear", "max_iter": 1000, "random_state": 0},
        },
        "length_baselines": {
            "char_length_auroc_harm_score_equals_longer": float(char_auc),
            "word_length_auroc_harm_score_equals_longer": float(word_auc),
            "mean_harm_chars": statistics.mean([row["harm_chars"] for row in per_pair]),
            "mean_safe_chars": statistics.mean([row["safe_chars"] for row in per_pair]),
            "mean_delta_chars_safe_minus_harm": statistics.mean([row["delta_chars_safe_minus_harm"] for row in per_pair]),
            "mean_harm_words": statistics.mean([row["harm_words"] for row in per_pair]),
            "mean_safe_words": statistics.mean([row["safe_words"] for row in per_pair]),
            "mean_delta_words_safe_minus_harm": statistics.mean([row["delta_words_safe_minus_harm"] for row in per_pair]),
        },
        "exact_matches": {"count": len(exact), "ids": exact},
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    report = [
        "# In-the-Wild Pair Text Baseline Audit (sklearn)",
        "",
        f"- Pairs: `{len(pairs)}`",
        f"- TF-IDF centroid LOPO AUROC: `{centroid_auc:.4f}`",
        f"- TF-IDF logistic LOPO AUROC: `{logreg_auc:.4f}`",
        f"- Length-only char AUROC (longer = harm): `{char_auc:.4f}`",
        f"- Length-only word AUROC (longer = harm): `{word_auc:.4f}`",
        f"- Exact matches: `{len(exact)}`",
        "",
        "This is a separate fast sklearn audit, not the standard pure-Python audit.",
        "",
    ]
    if args.note:
        report.extend(["## Note", "", args.note, ""])
    report.extend(
        [
            "## Length Summary",
            "",
            f"- Mean harm chars: `{metrics['length_baselines']['mean_harm_chars']:.1f}`",
            f"- Mean safe chars: `{metrics['length_baselines']['mean_safe_chars']:.1f}`",
            f"- Mean safe-harm delta chars: `{metrics['length_baselines']['mean_delta_chars_safe_minus_harm']:.1f}`",
            f"- Mean harm words: `{metrics['length_baselines']['mean_harm_words']:.1f}`",
            f"- Mean safe words: `{metrics['length_baselines']['mean_safe_words']:.1f}`",
            f"- Mean safe-harm delta words: `{metrics['length_baselines']['mean_delta_words_safe_minus_harm']:.1f}`",
            "",
            "## Top Safe Features",
            "",
        ]
    )
    for feature, coef in top_safe[:25]:
        report.append(f"- `{feature}`: `{coef:.4f}`")
    report.extend(["", "## Top Harm Features", ""])
    for feature, coef in top_harm[:25]:
        report.append(f"- `{feature}`: `{coef:.4f}`")
    (out_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "out_dir": args.out_dir,
                "n_pairs": len(pairs),
                "centroid_auc": round(float(centroid_auc), 4),
                "logreg_auc": round(float(logreg_auc), 4),
                "length_char_auc": round(float(char_auc), 4),
                "length_word_auc": round(float(word_auc), 4),
                "exact_matches": exact,
                "top_safe_features": [feature for feature, _ in top_safe[:10]],
                "top_harm_features": [feature for feature, _ in top_harm[:10]],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
