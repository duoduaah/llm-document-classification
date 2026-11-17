
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ENCODERS = {"emb_en_lr": "sentence-transformers/all-MiniLM-L6-v2",
            "emb_multi_lr": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dir", default="data/experiments")
    ap.add_argument("--no-embeddings", action="store_true")
    a = ap.parse_args()
    E = lambda n: os.path.join(a.exp_dir, n)
    tr = pd.read_parquet(E("exp3_train_en_chunks.parquet")).dropna(subset=["text"])
    A = pd.read_parquet(E("exp3_partA.parquet")).assign(part="A")
    B = pd.read_parquet(E("exp3_partB.parquet")).assign(part="B", condition="freeform")
    te = pd.concat([A, B], ignore_index=True)

    preds = {}
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=50000)
    clf = LogisticRegression(C=10, max_iter=3000).fit(vec.fit_transform(tr.text), tr.label)
    preds["tfidf_lr"] = clf.predict(vec.transform(te.text))

    if not a.no_embeddings:
        from sentence_transformers import SentenceTransformer
        for name, model_id in ENCODERS.items():
            print(f"encoding with {model_id} (first run downloads the model) ...")
            m = SentenceTransformer(model_id)
            enc = lambda s: m.encode(list(s), batch_size=32, normalize_embeddings=True, show_progress_bar=True)
            clf = LogisticRegression(C=10, max_iter=3000).fit(enc(tr.text), tr.label)
            preds[name] = clf.predict(enc(te.text))

    rows = []
    for meth, p in preds.items():
        t = te.assign(pred=p)
        for (cond, lg), g in t[t.part == "A"].groupby(["condition", "lang"]):
            rows.append({"method": meth, "part": "A", "condition": cond, "lang": lg, "n": len(g),
                         "legal_recall": round((g.pred == "legal_document").mean(), 3)})
        for lg, g in t[t.part == "B"].groupby("lang"):
            rows.append({"method": meth, "part": "B", "condition": "freeform", "lang": lg, "n": len(g),
                         "accuracy": round((g.pred == g.label).mean(), 3)})
    s = pd.DataFrame(rows)
    s.to_csv(E("exp3_baselines_summary.csv"), index=False)
    print("\n== Part A: legal recall on the SAME laws, by language (trained on English only)")
    print(s[s.part == "A"].pivot_table(index=["method", "condition"], columns="lang", values="legal_recall").to_string())
    if (s.part == "B").any():
        print("\n== Part B: accuracy, legal vs books (5-class model, so any other prediction counts as wrong)")
        print(s[s.part == "B"].pivot_table(index="method", columns="lang", values="accuracy").to_string())
    print(f"\nwrote {E('exp3_baselines_summary.csv')}")


if __name__ == "__main__":
    main()
