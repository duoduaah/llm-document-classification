
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score

TRAIN_VARIANTS = ["structured_first", "stripped_random"]


# ----------------------------------------------------------------- features
class TfidfModel:
    name = "tfidf_lr"

    def __init__(self, C):
        self.vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=50000)
        self.clf = LogisticRegression(C=C, max_iter=2000)

    def fit(self, df, E=None):
        self.clf.fit(self.vec.fit_transform(df.text), df.label)
        return self

    def proba(self, df, E=None):
        return self.clf.predict_proba(self.vec.transform(df.text))


class EmbModel:
    name = "emb_lr"

    def __init__(self, C):
        self.clf = LogisticRegression(C=C, max_iter=3000)

    def fit(self, df, E):
        self.clf.fit(E.take(df.item_id), df.label)
        return self

    def proba(self, df, E):
        return self.clf.predict_proba(E.take(df.item_id))


class EmbeddingStore:
    """Sentence embeddings for every variant text, computed once and cached."""

    def __init__(self, item_ids, texts, model_name, cache_path):
        self.index = {i: k for k, i in enumerate(item_ids)}
        if os.path.exists(cache_path):
            z = np.load(cache_path, allow_pickle=True)
            if list(z["ids"]) == list(item_ids):
                self.E = z["E"]
                print(f"loaded embedding cache {cache_path}")
                return
        from sentence_transformers import SentenceTransformer
        print(f"embedding {len(texts)} texts with {model_name} (CPU; a few minutes) ...")
        m = SentenceTransformer(model_name)
        print(f"  note: this encoder truncates input at {m.max_seq_length} word pieces, so LONG windows "
              f"above roughly {int(m.max_seq_length * 0.75)} words are effectively cut.")
        self.E = m.encode(list(texts), batch_size=64, normalize_embeddings=True, show_progress_bar=True)
        np.savez(cache_path, ids=np.array(item_ids, dtype=object), E=self.E)

    def take(self, ids):
        return self.E[[self.index[i] for i in ids]]


# ------------------------------------------------------------------ helpers
def fit_select(Model, Cs, tr, va, E):
    """Train for each C, keep the one with the best macro-F1 on the val split."""
    best = None
    for C in Cs:
        m = Model(C).fit(tr, E)
        pv = m.clf.classes_[m.proba(va, E).argmax(1)]
        f = f1_score(va.label, pv, average="macro")
        if best is None or f > best[0]:
            best = (f, C, m)
    return best[2], best[1]


def bootstrap_ci(y, p, n, seed):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y), (n, len(y)))
    vals = [f1_score(y[i], p[i], average="macro") for i in idx]
    return np.percentile(vals, [2.5, 97.5])


def margin_of(P):
    s = np.sort(P, axis=1)
    return s[:, -1] - s[:, -2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--no-embeddings", action="store_true")
    ap.add_argument("--emb-model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--boot", type=int, default=500, help="bootstrap resamples for the CI")
    ap.add_argument("--Cs", type=float, nargs="*", default=[0.1, 1, 10])
    ap.add_argument("--margin-model", default="tfidf_lr", choices=["tfidf_lr", "emb_lr"])
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    seed, exp = cfg["seed"], cfg["paths"]["experiments"]
    os.makedirs(exp, exist_ok=True)

    v = pd.read_parquet(os.path.join(exp, "exp1_variants.parquet")).reset_index(drop=True)
    models = [TfidfModel]
    E = None
    if not args.no_embeddings:
        E = EmbeddingStore(v.item_id.tolist(), v.text.tolist(), args.emb_model,
                           os.path.join(exp, "emb_cache.npz"))
        models.append(EmbModel)

    results, margin_rows, pred_rows, abst_rows = [], [], {}, []
    t0 = time.time()
    for (regime, window), g in v.groupby(["regime", "window"]):
        for train_variant in TRAIN_VARIANTS:
            tr = g[(g.split == "train") & (g.variant == train_variant)]
            va = g[(g.split == "val") & (g.variant == train_variant)]
            if tr.label.nunique() < 2 or va.empty:
                continue
            for Model in models:
                m, C = fit_select(Model, args.Cs, tr, va, E)
                classes = m.clf.classes_
                # abstention on the out-of-scope probe set: threshold keeps 95% of in-scope val items
                pr = g[(g.split == "probe") & (g.variant == train_variant)]
                ti = g[(g.split == "test") & (g.variant == train_variant)]
                if len(pr) and len(ti):
                    tau = float(np.quantile(m.proba(va, E).max(1), 0.05))
                    c_in, c_pr = m.proba(ti, E).max(1), m.proba(pr, E).max(1)
                    auc = roc_auc_score([0] * len(c_in) + [1] * len(c_pr), -np.r_[c_in, c_pr])
                    abst_rows.append({"model": Model.name, "regime": regime, "window": window,
                                      "train_variant": train_variant, "scope": "ALL probe", "n": len(pr),
                                      "abstain_rate": round(float((c_pr < tau).mean()), 3),
                                      "false_abstain_in_scope": round(float((c_in < tau).mean()), 3),
                                      "auroc": round(float(auc), 3), "tau": round(tau, 3)})
                    for src, gp in pr.groupby("source"):
                        abst_rows.append({"model": Model.name, "regime": regime, "window": window,
                                          "train_variant": train_variant, "scope": src, "n": len(gp),
                                          "abstain_rate": round(float((m.proba(gp, E).max(1) < tau).mean()), 3),
                                          "false_abstain_in_scope": round(float((c_in < tau).mean()), 3),
                                          "auroc": float("nan"), "tau": round(tau, 3)})
                for test_variant in sorted(g.variant.unique()):
                    te = g[(g.split == "test") & (g.variant == test_variant)]
                    if te.empty:
                        continue
                    P = m.proba(te, E)
                    pred = classes[P.argmax(1)]
                    y = te.label.to_numpy(dtype=object)
                    lo, hi = bootstrap_ci(y, pred, args.boot, seed)
                    results.append({"model": Model.name, "regime": regime, "window": window, "C": C,
                                    "train_variant": train_variant, "test_variant": test_variant,
                                    "n_test": len(te),
                                    "macro_f1": round(f1_score(y, pred, average="macro"), 4),
                                    "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
                                    "accuracy": round(accuracy_score(y, pred), 4)})
                    # margins + weak-worker predictions: SHORT regime, trained on stripped text
                    if (Model.name == args.margin_model and regime == "short"
                            and train_variant == "stripped_random" and test_variant != "structured_first"):
                        mar = margin_of(P)
                        margin_rows.extend({"item_id": i, "margin": float(x)} for i, x in zip(te.item_id, mar))
                        pred_rows.setdefault(Model.name, []).extend(
                            {"item_id": i, "label": str(pl), "confidence": float(p.max()),
                             "evidence": f"baseline {Model.name}", "ambiguous": bool(mg < 0.15),
                             "runner_up": str(classes[np.argsort(p)[-2]])}
                            for i, pl, p, mg in zip(te.item_id, pred, P, mar))
                    if (Model.name == "tfidf_lr" and regime == "short" and train_variant == test_variant
                            == "stripped_random"):
                        print(f"\n== per-class report: tfidf_lr, SHORT, train=test=stripped_random (C={C})")
                        print(classification_report(y, pred, zero_division=0))
        print(f"  done {regime}/{window}  ({time.time() - t0:.0f}s)")

    res = pd.DataFrame(results)
    res.to_csv(os.path.join(exp, "baseline_results.csv"), index=False)
    mat = res.pivot_table(index=["model", "regime", "window", "train_variant"], columns="test_variant",
                          values="macro_f1").round(3)
    mat.to_csv(os.path.join(exp, "baseline_matrix.csv"))
    print("\n== macro-F1 matrix (rows: model/regime/window/train variant; columns: test variant)")
    print(mat.to_string())
    if abst_rows:
        ab = pd.DataFrame(abst_rows)
        ab.to_csv(os.path.join(exp, "baseline_abstention.csv"), index=False)
        print("\n== abstention on the out-of-scope probe set (baselines cannot say 'other'; a confidence threshold is used)")
        print(ab[ab.train_variant == "stripped_random"].to_string(index=False))
    if margin_rows:
        pd.DataFrame(margin_rows).to_csv(os.path.join(exp, "margins.csv"), index=False)
        print(f"\nwrote margins.csv ({len(margin_rows)} rows) -> use with: build_experiments.py gold-hard --margins")
    for name, rows in pred_rows.items():
        p = os.path.join(exp, f"preds_baseline_{name}.jsonl")
        with open(p, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        print(f"wrote {p} ({len(rows)} predictions; weak second worker for the judge experiment)")
    print("\nHow to read the matrix:")
    print("  - train=structured_first vs test=stripped_random: the drop is how much the model leans on structure (Q1).")
    print("  - test noise2/5/10: fragility to OCR-like errors.")
    print("  - Look at ci_lo/ci_hi in baseline_results.csv before claiming a difference; small test sets give wide intervals.")


if __name__ == "__main__":
    main()
