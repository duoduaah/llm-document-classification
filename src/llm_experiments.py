import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from docclf import CLASSES  
from llm_runner import Runner  

LABELS = CLASSES + ["other"]


def macro_f1_ci(y, p, n_boot=500, seed=13):
    y, p = np.asarray(y, dtype=object), np.asarray(p, dtype=object)
    labs = sorted(set(y))                                   
    f = f1_score(y, p, labels=labs, average="macro", zero_division=0)
    rng = np.random.default_rng(seed)
    bs = [f1_score(y[i], p[i], labels=labs, average="macro", zero_division=0)
          for i in rng.integers(0, len(y), (n_boot, len(y)))]
    return f, *np.percentile(bs, [2.5, 97.5])


def run(df, a, text_col="text"):
    r = Runner(a.config, a.model)
    if a.dry_run:
        r.dry = []
    preds = r.predict(df[text_col].tolist())
    if a.dry_run:
        n, tin = len(r.dry), sum(r.dry)
        cost = ""
        if r.mc.get("price_in") is not None:
            cost = f", about ${tin / 1e6 * r.mc['price_in'] + n * 90 / 1e6 * r.mc.get('price_out', 0):.2f}"
        print(f"DRY RUN: {n} new calls on {r.mc['id']} ({tin:,} input tokens{cost}). "
              f"At {r.mc['rpm']} rpm: about {n / (r.mc['rpm'] / r.cfg['safety_margin']):.0f} min.")
        return None, r
    out = df.drop(columns=[text_col]).copy()
    for k in ("label", "confidence", "ambiguous", "runner_up"):
        out["pred_" + k] = [p.get(k) for p in preds]
    out["invalid"] = [bool(p.get("invalid")) for p in preds]
    return out, r


# ------------------------------------------------------------------ Experiment 1
def cmd_exp1(a):
    v = pd.read_parquet(os.path.join(a.exp_dir, "exp1_variants.parquet"))
    v = v[v.split == "test"]
    docs = (v.drop_duplicates("doc_id").groupby("label", group_keys=False)
            .apply(lambda g: g.sample(min(a.per_class, len(g)), random_state=13)).doc_id)
    v = v[v.doc_id.isin(set(docs))]
    v = v[(v.regime == "short") | v.window.isin(a.long_windows)]
    print(f"Experiment 1: {v.doc_id.nunique()} test documents, {len(v)} variant items")
    out, r = run(v, a)
    if out is None:
        return
    name = r.mc["id"]
    out.to_csv(os.path.join(a.exp_dir, f"llm_exp1_{name}.csv"), index=False)
    rows = []
    for (reg, w, var), g in out.groupby(["regime", "window", "variant"]):
        f, lo, hi = macro_f1_ci(g.label, g.pred_label)
        rows.append({"regime": reg, "window": w, "variant": var, "n": len(g),
                     "macro_f1": round(f, 3), "ci_lo": round(lo, 3), "ci_hi": round(hi, 3),
                     "accuracy": round((g.label == g.pred_label).mean(), 3),
                     "said_other": round((g.pred_label == "other").mean(), 3),
                     "said_ambiguous": round(g.pred_ambiguous.mean(), 3),
                     "invalid": int(g.invalid.sum())})
    s = pd.DataFrame(rows)
    s.to_csv(os.path.join(a.exp_dir, f"llm_exp1_{name}_summary.csv"), index=False)
    print(f"\n== Experiment 1, LLM worker {name} (zero-shot; truth = source label)")
    print(s.to_string(index=False))
    sh = out[(out.regime == "short") & (out.variant == "stripped_random")]
    print("\n== SHORT stripped_random: which class gets confused with which (rows = truth)")
    print(pd.crosstab(sh.label, sh.pred_label).to_string())
    print("\nCompare with baseline_matrix.csv (TF-IDF trained on stripped_random). Baseline scores use the full "
          "test split; the LLM uses a sample, so compare the confidence intervals, not just the point values.")


# ---------------------------------------------------------------------- probe
def cmd_probe(a):
    v = pd.read_parquet(os.path.join(a.exp_dir, "exp1_variants.parquet"))
    pr = v[(v.split == "probe") & (v.regime == "short") & (v.variant == "stripped_random")]
    print(f"Probe: {len(pr)} out-of-scope items (SHORT, stripped)")
    out, r = run(pr, a)
    if out is None:
        return
    name = r.mc["id"]
    out.to_csv(os.path.join(a.exp_dir, f"llm_probe_{name}.csv"), index=False)
    print(f"\n== out-of-scope probe, {name}: does it say 'other' when nothing fits?")
    g = out.groupby("source").agg(n=("pred_label", "size"),
                                  said_other=("pred_label", lambda s: round((s == "other").mean(), 3)),
                                  said_ambiguous=("pred_ambiguous", "mean"))
    print(g.round(3).to_string())
    print(f"ALL probe: said other = {(out.pred_label == 'other').mean():.3f}")
    print("\nWhat it called the probe items instead (rows = probe source):")
    print(pd.crosstab(out.source, out.pred_label).to_string())
    f = os.path.join(a.exp_dir, f"llm_exp1_{name}.csv")
    if os.path.exists(f):
        e1 = pd.read_csv(f)
        e1 = e1[(e1.regime == "short") & (e1.variant == "stripped_random")]
        print(f"\nfalse 'other' on in-scope SHORT stripped items: {(e1.pred_label == 'other').mean():.3f} "
              f"(n={len(e1)}). Baselines needed a confidence threshold for this; the LLM can abstain directly.")


# ----------------------------------------------------------------- Experiment 3
def cmd_exp3(a):
    A = pd.read_parquet(os.path.join(a.exp_dir, "exp3_partA.parquet")).assign(part="A")
    B = pd.read_parquet(os.path.join(a.exp_dir, "exp3_partB.parquet")).assign(part="B", condition="freeform")
    df = pd.concat([A, B], ignore_index=True)
    print(f"Experiment 3: Part A {len(A)} rows, Part B {len(B)} rows")
    out, r = run(df, a)
    if out is None:
        return
    name = r.mc["id"]
    out.to_csv(os.path.join(a.exp_dir, f"llm_exp3_{name}.csv"), index=False)
    rows = []
    oa = out[out.part == "A"]
    for (cond, lg), g in oa.groupby(["condition", "lang"]):
        rows.append({"part": "A", "condition": cond, "lang": lg, "n": len(g),
                     "legal_recall": round((g.pred_label == "legal_document").mean(), 3),
                     "said_other": round((g.pred_label == "other").mean(), 3),
                     "mean_conf": round(g.pred_confidence.mean(), 3)})
    ob = out[out.part == "B"]
    for lg, g in ob.groupby("lang"):
        f, lo, hi = macro_f1_ci(g.label, g.pred_label) if g.label.nunique() > 1 else (np.nan,) * 3
        rows.append({"part": "B", "condition": "freeform", "lang": lg, "n": len(g),
                     "accuracy": round((g.label == g.pred_label).mean(), 3),
                     "macro_f1": round(f, 3), "ci_lo": round(lo, 3), "ci_hi": round(hi, 3),
                     "said_other": round((g.pred_label == "other").mean(), 3),
                     "mean_conf": round(g.pred_confidence.mean(), 3)})
    s = pd.DataFrame(rows)
    s.to_csv(os.path.join(a.exp_dir, f"llm_exp3_{name}_summary.csv"), index=False)
    print(f"\n== Experiment 3, LLM worker {name} (English instructions, any-language input)")
    print(s.to_string(index=False))
    if len(oa):
        print("\nPart A: what the SAME laws were called, by language (rows = language)")
        print(pd.crosstab([oa.condition, oa.lang], oa.pred_label).to_string())
    print("\nPart A reads as: same 70 laws in every language, so any gap between 'en' and another language is "
          "caused by the language alone.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["exp1", "probe", "exp3"])
    ap.add_argument("--config", default="configs/llm.yaml")
    ap.add_argument("--model", default="c", choices=["a", "b", "c"])
    ap.add_argument("--exp-dir", default="data/experiments")
    ap.add_argument("--per-class", type=int, default=30, help="Exp 1: test documents per class")
    ap.add_argument("--long-windows", type=int, nargs="*", default=[128, 256, 1024],
                    help="Exp 1: LONG windows to run (each adds items)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    {"exp1": cmd_exp1, "probe": cmd_probe, "exp3": cmd_exp3}[a.cmd](a)


if __name__ == "__main__":
    main()
