
import argparse
import json
import os
import sys
from collections import defaultdict

import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(__file__))
from docclf import (CLASSES, artifact_features, collapse, generic_strip_structure,  # noqa: E402
                    n_tokens, ocr_noise, seeded_rng, window_chars, window_tokens)


def load_cfg(path):
    return yaml.safe_load(open(path))


def load_docs(cfg):
    return pd.read_parquet(cfg["paths"]["processed"])


def exp_path(cfg, name):
    os.makedirs(cfg["paths"]["experiments"], exist_ok=True)
    return os.path.join(cfg["paths"]["experiments"], name)


def gold_path(cfg, name):
    os.makedirs(cfg["paths"]["gold"], exist_ok=True)
    return os.path.join(cfg["paths"]["gold"], name)


# ======================================================== Experiment 1 (Q1)
def cmd_exp1(cfg, args):
    e, seed = cfg["exp1"], cfg["seed"]
    docs = load_docs(cfg)
    docs = docs[((docs.role == "main") & docs.split.isin(["train", "val", "test"])) | (docs.split == "probe")]
    long_classes = [c for c in CLASSES if c != "news_article"]       # news is too short
    rows, dropped = [], defaultdict(int)

    def add(r, regime, w, variant, text):
        rows.append({"item_id": f"{r.doc_id}|{regime}|{w}|{variant}", "doc_id": r.doc_id,
                     "split": r.split, "label": r.label, "source": r.source, "regime": regime,
                     "window": w, "variant": variant, "text": text, "n_tokens": n_tokens(text)})

    def make(r, regime, w, variant, base, mode):
        t = window_tokens(base, w, mode, seeded_rng(seed, r.doc_id, regime, w, variant))
        if t is None:
            dropped[(regime, w, variant, r.label)] += 1
            return None
        add(r, regime, w, variant, t)
        return t

    def noisy(r, regime, w, base_text):
        for p in e["noise_rates"]:
            add(r, regime, w, f"noise{int(round(p * 100))}",
                ocr_noise(base_text, p, seeded_rng(seed, r.doc_id, regime, w, p)))

    for r in docs.itertuples():
        structured, stripped = collapse(r.text_norm), generic_strip_structure(r.text_norm)
        w = e["short_window"]                                          # SHORT regime: all classes
        make(r, "short", w, "structured_first", structured, "first")
        t = make(r, "short", w, "stripped_random", stripped, "random")
        if t and r.split == "test":
            noisy(r, "short", w, t)
        if r.label in long_classes or r.split == "probe":              # LONG regime
            for w in e["long_windows"]:
                make(r, "long", w, "structured_first", structured, "first")
                t = make(r, "long", w, "stripped_random", stripped, "random")
                if t and w == e["noise_window_long"] and r.split == "test":
                    noisy(r, "long", w, t)
            make(r, "long", e["sme_window"], "stripped_sme", stripped, "sme")

    out = pd.DataFrame(rows)
    out.to_parquet(exp_path(cfg, "exp1_variants.parquet"), index=False)
    pd.DataFrame([{"regime": k[0], "window": k[1], "variant": k[2], "label": k[3], "dropped": v}
                  for k, v in dropped.items()]).to_csv(exp_path(cfg, "exp1_dropped.csv"), index=False)
    print(f"wrote {len(out)} variant rows -> {exp_path(cfg, 'exp1_variants.parquet')}")
    print("\ntest-split items per regime/window/variant x label")
    t = out[out.split == "test"]
    print(pd.crosstab([t.regime, t.window, t.variant], t.label).to_string())
    if dropped:
        print(f"\n{sum(dropped.values())} (doc, variant) pairs dropped for being shorter than the window; "
              f"see exp1_dropped.csv")
    # retention check: a class that loses many documents to the window is a bias risk
    tot = docs[docs.split == "test"].groupby("label").size()
    kept = out[(out.split == "test") & (out.regime == "short") & (out.variant == "stripped_random")].groupby("label").size()
    ret = pd.DataFrame({"test_docs": tot, "kept_short": kept}).fillna(0).astype(int)
    ret["retention"] = (ret.kept_short / ret.test_docs).round(2)
    print("\nSHORT-regime retention per class (test split):")
    print(ret.to_string())
    pr = docs[docs.split == "probe"].groupby("source").size()
    prk = out[(out.split == "probe") & (out.regime == "short") & (out.variant == "stripped_random")].groupby("source").size()
    if len(pr):
        print("\nPROBE (out-of-scope) items kept in the SHORT regime, by source:")
        print(pd.DataFrame({"probe_docs": pr, "kept_short": prk}).fillna(0).astype(int).to_string())
    for lab, r in ret.iterrows():
        if r.retention < 0.8:
            print(f"WARNING: {lab} keeps only {r.retention:.0%} of its test documents in the SHORT regime. "
                  f"Lower exp1.short_window in configs/data.yaml (short sources such as ag_news need this).")


# -------------------------------------------------------------- leakage check
def cmd_leakage(cfg, args):
    """Can artifact-only features (no words) predict the class?  If yes, a
    model could learn extraction/format shortcuts instead of content."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import f1_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    docs = load_docs(cfg)
    docs = docs[(docs.role == "main") & docs.split.isin(["train", "val", "test"])].reset_index(drop=True)
    chance = 1 / docs.label.nunique()
    checks = [("normalized text WITH line structure (models never see this)",
               pd.DataFrame([artifact_features(t) for t in docs.text_norm]).astype(float),
               docs.label.to_numpy(dtype=object)),
              ("length only: n_tokens of whole document",
               docs[["n_tokens"]].astype(float), docs.label.to_numpy(dtype=object))]
    vp = exp_path(cfg, "exp1_variants.parquet")
    if os.path.exists(vp):
        v = pd.read_parquet(vp)
        v = v[v.split != "probe"]
        for variant in ("structured_first", "stripped_random"):
            g = v[(v.regime == "short") & (v.variant == variant)]
            checks.append((f"SHORT windows, {variant} (what models see)",
                           pd.DataFrame([artifact_features(t) for t in g.text]).astype(float),
                           g.label.to_numpy(dtype=object)))
    print(f"chance macro-F1 ~ {chance:.2f}. Values far above chance mean a shortcut remains.\n")
    for name, X, y in checks:
        pred = cross_val_predict(RandomForestClassifier(200, random_state=cfg["seed"], n_jobs=-1), X, y,
                                 cv=StratifiedKFold(5, shuffle=True, random_state=cfg["seed"]))
        print(f"{name:66s} macro-F1 = {f1_score(y, pred, average='macro'):.3f}")
    print("\nHow to read this: the first two rows will usually be HIGH (PDF text keeps line breaks, other sources do not; "
          "documents differ in length). That is why every model input is flattened and cut to equal windows. "
          "The SHORT-window rows are the ones that must sit near chance; if they do not, tighten normalization.")


# ============================================================== Gold set (Q2)
def _write_gold(cfg, pick, tag):
    sheet = pick[["item_id", "text"]].sample(frac=1, random_state=cfg["seed"]).copy()   # shuffle classes
    sheet["my_label"], sheet["ambiguous"], sheet["notes"] = "", "", ""
    sheet.to_csv(gold_path(cfg, f"gold_{tag}_annotation.csv"), index=False)
    pick[["item_id", "doc_id", "label"]].rename(columns={"label": "source_label"}).to_csv(
        gold_path(cfg, f"gold_{tag}_key.csv"), index=False)
    print(f"wrote gold_{tag}_annotation.csv ({len(sheet)} items, labels hidden) and gold_{tag}_key.csv (keep closed while labeling)")


def cmd_gold_random(cfg, args):
    v = pd.read_parquet(exp_path(cfg, "exp1_variants.parquet"))
    pool = v[(v.regime == "short") & (v.variant == "stripped_random") & (v.split == "test")]
    n = cfg["exp2"]["gold_random_per_class"]
    parts = []
    for c, g in pool.groupby("label"):
        if len(g) < n:
            print(f"warning: only {len(g)} test items for {c}")
        parts.append(g.sample(min(n, len(g)), random_state=cfg["seed"]))
    _write_gold(cfg, pd.concat(parts), "random")


def cmd_gold_hard(cfg, args):
    v = pd.read_parquet(exp_path(cfg, "exp1_variants.parquet"))
    margins = pd.read_csv(args.margins)                                  # item_id, margin (low = hard)
    used = set(pd.read_csv(gold_path(cfg, "gold_random_key.csv")).doc_id)
    pool = v[(v.regime == "short") & v.variant.isin(["stripped_random", "noise10"]) & (v.split == "test")]
    pool = pool.merge(margins, on="item_id").query("doc_id not in @used").sort_values("margin")
    cap, total = cfg["exp2"]["gold_hard_per_class_cap"], cfg["exp2"]["gold_hard_total"]
    per, picked, seen_docs = defaultdict(int), [], set()
    for r in pool.itertuples():
        if per[r.label] >= cap or r.doc_id in seen_docs:
            continue
        per[r.label] += 1
        seen_docs.add(r.doc_id)
        picked.append(r.Index)
        if len(picked) >= total:
            break
    _write_gold(cfg, pool.loc[picked], "hard")


def cmd_gold_relabel(cfg, args):
    frames = [pd.read_csv(p) for p in args.labels]
    lab = pd.concat(frames)
    lab = lab[lab.my_label.notna() & (lab.my_label.astype(str).str.strip() != "")]
    pick = lab.sample(min(cfg["exp2"]["relabel_n"], len(lab)), random_state=cfg["seed"] + 1)
    sheet = pick[["item_id", "text"]].copy()
    sheet["my_label"], sheet["ambiguous"], sheet["notes"] = "", "", ""
    sheet.to_csv(gold_path(cfg, "gold_relabel_annotation.csv"), index=False)
    print(f"wrote gold_relabel_annotation.csv ({len(sheet)} items). Label it a day after the first pass, "
          f"without looking at your earlier labels.")


def cmd_probe_sample(cfg, args):
    """Dump 30 random probe documents so you can confirm none belongs to the five classes."""
    d = load_docs(cfg)
    d = d[d.split == "probe"]
    pick = d.sample(min(30, len(d)), random_state=cfg["seed"])
    sheet = pd.DataFrame({"doc_id": pick.doc_id, "source": pick.source,
                          "snippet": pick.text_norm.str.slice(0, 500),
                          "belongs_to_one_of_5_classes (y/n)": ""})
    sheet.to_csv(gold_path(cfg, "probe_check.csv"), index=False)
    print(f"wrote probe_check.csv ({len(sheet)} items). Fill the last column; any 'y' should be removed from the probe set.")


# ======================================================== placeholder + Exp 2
def cmd_placeholder_preds(cfg, args):
    from placeholder_models import get_worker
    v = pd.read_parquet(exp_path(cfg, "exp1_variants.parquet"))
    v = v[(v.split == "test") & (v.regime == args.regime)]
    if args.variants:
        v = v[v.variant.isin(args.variants)]
    preds = get_worker("placeholder").predict(v.text.tolist())
    with open(args.out, "w", encoding="utf-8") as fh:
        for item_id, p in zip(v.item_id, preds):
            fh.write(json.dumps({"item_id": item_id, **p}) + "\n")
    print(f"wrote {len(preds)} placeholder predictions -> {args.out}")


def cmd_exp2_judge(cfg, args):
    e, seed = cfg["exp2"], cfg["seed"]
    v = pd.read_parquet(exp_path(cfg, "exp1_variants.parquet")).set_index("item_id")
    key = pd.read_csv(args.key).set_index("item_id")
    truth = key.source_label.to_dict()
    if args.gold:                                                        
        g = pd.read_csv(args.gold)
        g = g[g.my_label.notna() & (g.my_label.astype(str).str.strip() != "")]
        truth.update(dict(zip(g.item_id, g.my_label.str.strip())))
    preds = {}
    for line in open(args.preds, encoding="utf-8"):
        d = json.loads(line)
        preds[d["item_id"]] = d["label"]
    ids = [i for i in truth if i in v.index]
    rng = seeded_rng(seed, "judge")
    corrupt = set(rng.sample(ids, int(round(e["corrupt_fraction"] * len(ids)))))
    bias_ids = set(rng.sample(ids, min(e["bias_items"], len(ids))))
    reqs = []

    def emit(item, cond, proposed, order, rubric):
        reqs.append({"request_id": f"{item}|{cond}|{order}|{'rubric' if rubric else 'norubric'}",
                     "item_id": item, "condition": cond, "rubric_order": order, "with_rubric": rubric,
                     "proposed_label": proposed, "truth_label": truth[item],
                     "proposed_is_correct": proposed == truth[item], "text": v.loc[item, "text"]})

    for i in ids:
        if i in preds:
            emit(i, "real", preds[i], "original", True)
        wrong = rng.choice([c for c in CLASSES if c != truth[i]]) if i in corrupt else truth[i]
        emit(i, "synthetic", wrong, "original", True)
        if i in bias_ids:                                                  # bias-test variants
            for cond, lab in (("real", preds.get(i)), ("synthetic", wrong)):
                if lab is not None:
                    emit(i, cond, lab, "shuffled", True)
                    emit(i, cond, lab, "original", False)
    out = exp_path(cfg, "judge_requests.jsonl")
    with open(out, "w", encoding="utf-8") as fh:
        for r in reqs:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    df = pd.DataFrame(reqs)
    print(f"wrote {len(df)} judge requests -> {out}")
    print(df.groupby(["condition", "rubric_order", "with_rubric"]).agg(
        n=("item_id", "size"), wrong=("proposed_is_correct", lambda s: int((~s).sum()))).to_string())
    n_real_err = int(((df.condition == "real") & (~df.proposed_is_correct) & (df.rubric_order == "original")
                      & df.with_rubric).sum())
    print(f"\nreal worker errors available to the judge: {n_real_err} (target >= ~60)")


def cmd_placeholder_judge(cfg, args):
    from placeholder_models import get_judge
    reqs = [json.loads(line) for line in open(exp_path(cfg, "judge_requests.jsonl"), encoding="utf-8")]
    res = get_judge("placeholder").verify([r["text"] for r in reqs], [r["proposed_label"] for r in reqs])
    out = exp_path(cfg, "judge_responses.jsonl")
    with open(out, "w", encoding="utf-8") as fh:
        for r, d in zip(reqs, res):
            fh.write(json.dumps({"request_id": r["request_id"], **d}) + "\n")
    df = pd.DataFrame(reqs).merge(pd.DataFrame(res).assign(request_id=[r["request_id"] for r in reqs]), on="request_id")
    base = df[(df.rubric_order == "original") & df.with_rubric]
    for cond, g in base.groupby("condition"):
        wrong, flagged = ~g.proposed_is_correct, g.verdict == "incorrect"
        tp = int((wrong & flagged).sum())
        print(f"[{cond}] error-detection recall={tp / max(int(wrong.sum()), 1):.2f} "
              f"precision={tp / max(int(flagged.sum()), 1):.2f} (PLACEHOLDER numbers, pipeline check only)")
    print(f"wrote {out}")


# ======================================================= Experiment 3 (Q3)
def cmd_exp3(cfg, args):
    e, seed = cfg["exp3"], cfg["seed"]
    N = e["chunk_chars"]
    docs = load_docs(cfg)

    # ---- English training chunks (same unit size as the multilingual test)
    tr = docs[(docs.role == "main") & docs.split.isin(["train"])]
    rows = []
    for r in tr.itertuples():
        t = window_chars(generic_strip_structure(r.text_norm), N, rng=seeded_rng(seed, r.doc_id, "train"))
        rows.append({"doc_id": r.doc_id, "label": r.label, "lang": "en", "text": t})
    pd.DataFrame(rows).to_parquet(exp_path(cfg, "exp3_train_en_chunks.parquet"), index=False)
    print(f"English training chunks: {len(rows)}")

    # ---- Part A: parallel legislation
    legal = docs[docs.role == "ml_legal"].copy()
    legal["celex"] = legal.meta.map(lambda m: json.loads(m).get("celex_id"))
    langs = e["languages_partA"]
    present = legal.groupby("celex")["lang_declared"].apply(set)
    complete = [c for c, s in present.items() if set(langs) <= s]
    print(f"\nPart A: {len(present)} celex ids; complete in all of {langs}: {len(complete)}")
    for lg in langs:
        print(f"  {lg}: {int(sum(lg in s for s in present))} documents")
    if len(complete) < len(present):
        print("  -> incomplete ids are excluded so every language sees the SAME documents "
              "(download the missing language files, e.g. English).")
    A = []
    for r in legal[legal.celex.isin(complete) & legal.lang_declared.isin(langs)].itertuples():
        A.append({"celex": r.celex, "lang": r.lang_declared, "condition": "start", "label": "legal_document",
                  "text": collapse(r.text_norm)[:N]})
        A.append({"celex": r.celex, "lang": r.lang_declared, "condition": "freeform", "label": "legal_document",
                  "text": window_chars(generic_strip_structure(r.text_norm), N, fraction=e["partA_fraction"])})
    pd.DataFrame(A).to_parquet(exp_path(cfg, "exp3_partA.parquet"), index=False)
    print(f"  wrote {len(A)} Part A rows")

    # ---- Part B: legal vs literature in it/fr/es
    B = []
    for lg in e["languages_partB"]:
        for role, label in (("ml_legal", "legal_document"), ("ml_books", "book_literature")):
            g = docs[(docs.role == role) & (docs.lang_declared == lg)]
            g = g.sample(min(e["partB_per_class"], len(g)), random_state=seed)
            if len(g) < e["partB_per_class"]:
                print(f"  warning: Part B {lg}/{label}: only {len(g)} documents")
            for r in g.itertuples():
                B.append({"doc_id": r.doc_id, "lang": lg, "label": label,
                          "text": window_chars(generic_strip_structure(r.text_norm), N,
                                               rng=seeded_rng(seed, r.doc_id, "partB"))})
    pd.DataFrame(B).to_parquet(exp_path(cfg, "exp3_partB.parquet"), index=False)
    print(f"Part B: wrote {len(B)} rows")
    print(pd.crosstab(pd.DataFrame(B).lang, pd.DataFrame(B).label).to_string())

    # ---- language-ID report
    ml = docs[docs.role.isin(["ml_legal", "ml_books"])]
    rep = ml.groupby(["lang_declared", "lang_detected"]).size().rename("n").reset_index()
    rep.to_csv(exp_path(cfg, "lang_id_report.csv"), index=False)
    agree = (ml.lang_declared == ml.lang_detected).mean() if len(ml) else float("nan")
    print(f"language-ID agreement on multilingual docs: {agree:.3f} (report -> lang_id_report.csv)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["exp1", "leakage", "gold-random", "gold-hard", "gold-relabel",
                                    "placeholder-preds", "exp2-judge", "placeholder-judge", "exp3", "probe-sample"])
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--margins")
    ap.add_argument("--labels", nargs="*")
    ap.add_argument("--out", default="data/experiments/preds.jsonl")
    ap.add_argument("--regime", default="short")
    ap.add_argument("--variants", nargs="*")
    ap.add_argument("--preds")
    ap.add_argument("--key")
    ap.add_argument("--gold")
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    {"exp1": cmd_exp1, "leakage": cmd_leakage, "gold-random": cmd_gold_random, "gold-hard": cmd_gold_hard,
     "gold-relabel": cmd_gold_relabel, "placeholder-preds": cmd_placeholder_preds, "exp2-judge": cmd_exp2_judge,
     "placeholder-judge": cmd_placeholder_judge, "exp3": cmd_exp3, "probe-sample": cmd_probe_sample}[args.cmd](cfg, args)


if __name__ == "__main__":
    main()
