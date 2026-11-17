
import argparse
import json
import os
import random

import pandas as pd


def load_jsonl(p):
    return [json.loads(ln) for ln in open(p, encoding="utf-8")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dir", default="data/experiments")
    ap.add_argument("--key", required=True, help="gold key (item_id, doc_id, source_label)")
    ap.add_argument("--gold", required=True, help="filled gold sheet (item_id, my_label, ambiguous, ...)")
    ap.add_argument("--worker", default="gpt-6-luna")
    ap.add_argument("--judges", nargs="*", default=["gpt-6-luna", "gemini-3.1-flash-lite"])
    a = ap.parse_args()
    E = lambda n: os.path.join(a.exp_dir, n)

    req = pd.DataFrame(load_jsonl(E("judge_requests.jsonl")))
    req = req[(req.rubric_order == "original") & req.with_rubric]
    worker = {d["item_id"]: d for d in load_jsonl(E(f"preds_{a.worker}.jsonl"))}
    key = pd.read_csv(a.key).drop_duplicates("item_id").set_index("item_id")
    gold = pd.read_csv(a.gold).drop_duplicates("item_id").set_index("item_id")

    real = req[req.condition == "real"].set_index("item_id")
    items = pd.DataFrame({"human": real.truth_label,
                          "worker": real.proposed_label,
                          "source": key.source_label.reindex(real.index)})
    items["worker_conf"] = [worker.get(i, {}).get("confidence") for i in items.index]
    amb = gold["ambiguous"].reindex(items.index) if "ambiguous" in gold else pd.Series(index=items.index, dtype=object)
    items["human_ambiguous"] = amb.astype(str).str.strip().str.lower().isin(["1", "true", "yes", "y", "x"])
    dis = items[items.worker != items.human]

    print(f"== the {len(dis)} items where the worker disagrees with your label")
    print(f"  worker agrees with the SOURCE label: {(dis.worker == dis.source).sum()}  "
          f"(you overrode the source, the LLM sided with the source)")
    print(f"  you marked ambiguous: {dis.human_ambiguous.sum()} of {len(dis)}  "
          f"(vs {items[items.worker == items.human].human_ambiguous.mean():.0%} on items where you agree)")
    print(f"  worker confidence on them: mean {dis.worker_conf.mean():.2f}")
    print("\n  your label -> worker label (counts)")
    print(pd.crosstab(dis.human, dis.worker).to_string())

    for j in a.judges:
        p = E(f"judge_responses_{j}.jsonl")
        if not os.path.exists(p):
            continue
        resp = {d["request_id"]: d for d in load_jsonl(p)}
        syn = req[req.condition == "synthetic"].copy()
        syn["verdict"] = [resp[r]["verdict"] for r in syn.request_id]
        syn["suggested"] = [resp[r].get("suggested_label") for r in syn.request_id]
        fa = syn[syn.proposed_is_correct & (syn.verdict == "incorrect")]       # flagged a CORRECT (=your) label
        on_dis = fa.item_id.isin(dis.index)
        to_worker = [s == items.worker.get(i) for s, i in zip(fa.suggested, fa.item_id)]
        print(f"\n== judge {j}: {len(fa)} false alarms on the synthetic set (it rejected YOUR label)")
        print(f"  on items where the worker also disagreed with you: {on_dis.sum()} of {len(fa)}")
        print(f"  judge's suggested label = the worker's label: {sum(to_worker)} of {len(fa)}")
        # re-score the REAL condition against each version of the truth (no new API calls)
        rl = req[req.condition == "real"].copy()
        rl["verdict"] = [resp[r]["verdict"] for r in rl.request_id]
        rl["source"] = key.source_label.reindex(rl.item_id).to_numpy()
        for name, col in (("your labels", "truth_label"), ("source labels", "source")):
            wrong, flag = rl.proposed_label != rl[col], rl.verdict == "incorrect"
            tp = int((wrong & flag).sum())
            print(f"  real worker labels scored against {name:13s}: worker accuracy={1 - wrong.mean():.2f}  "
                  f"wrong={int(wrong.sum())}  judge caught={tp} (recall {tp / max(int(wrong.sum()), 1):.2f})  "
                  f"false alarms={int((~wrong & flag).sum())}")

    rng = random.Random(13)
    rows = []
    for i, r in dis.iterrows():
        a_, b_ = (r.human, r.worker) if rng.random() < 0.5 else (r.worker, r.human)
        rows.append({"item_id": i, "text": gold.text.get(i, ""), "option_1": a_, "option_2": b_,
                     "my_decision": "", "notes": ""})
    out = E("adjudication_sheet.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    hid = E("adjudication_key.csv")
    dis[["human", "worker", "source"]].to_csv(hid)
    print(f"\nwrote {out}: for each item write option_1, option_2, both, or neither in my_decision.")
    print(f"      (who said which is in {hid}; keep it closed while you decide)")


if __name__ == "__main__":
    main()
