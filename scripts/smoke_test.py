"""End-to-end smoke test on SYNTHETIC data (no real datasets needed).

  python scripts/smoke_test.py

Creates fake raw files in each real format (PDF dir, jsonl, JSON dir, parquet,
per-language MultiEURLEX files), then runs adapters -> Spark prepare -> every
experiment builder.  If this passes, the code paths work; point
configs/data.yaml at your real files for the real run.
"""
import json
import os
import random
import subprocess
import sys
import tempfile

import pandas as pd
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
rng = random.Random(0)

VOCAB = {
    "research_paper": "we propose a novel method experiments dataset baseline results show model algorithm theorem".split(),
    "financial_filing": "revenue fiscal year net income shareholders risk factors company operating securities cash".split(),
    "legal_document": "the court held that plaintiff defendant shall pursuant appeal statute judgment hereby".split(),
    "news_article": "said on tuesday officials announced percent reuters reported yesterday the team".split(),
    "book_literature": "he said she had been once upon the old night her eyes heart chapter".split(),
}
FILLER = "the of and to in a is that for with as on by this are from or an be at it".split()
LANG_TEXT = {
    "en": "The council adopted this regulation on the common market and the commission shall report every year. ",
    "de": "Der Rat hat diese Verordnung über den gemeinsamen Markt angenommen und die Kommission berichtet jedes Jahr. ",
    "fr": "Le Conseil a adopté ce règlement relatif au marché commun et la Commission fait rapport chaque année. ",
    "es": "El Consejo adoptó este reglamento sobre el mercado común y la Comisión informará cada año. ",
    "it": "Il Consiglio ha adottato questo regolamento sul mercato comune e la Commissione riferisce ogni anno. ",
}
BOOK_TEXT = {
    "it": "La vecchia casa era silenziosa e la notte scendeva lentamente sul giardino mentre lei guardava il cielo. ",
    "fr": "La vieille maison était silencieuse et la nuit tombait lentement sur le jardin pendant qu'elle regardait le ciel. ",
    "es": "La vieja casa estaba en silencio y la noche caía lentamente sobre el jardín mientras ella miraba el cielo. ",
}


def words(cls, n):
    return " ".join(rng.choice(VOCAB[cls] if rng.random() < 0.35 else FILLER) for _ in range(n))


def wrap(text, width=78):
    out, line = [], ""
    for w in text.split():
        if len(line) + len(w) + 1 > width:
            out.append(line)
            line = w
        else:
            line = f"{line} {w}".strip()
    return "\n".join(out + [line])


def make_pdf(path, text):
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    c = canvas.Canvas(path, pagesize=letter)
    y = 750
    for ln in text.split("\n"):
        c.drawString(40, y, ln[:110])
        y -= 12
        if y < 60:
            c.drawString(300, 30, "7")
            c.showPage()
            y = 750
    c.save()


def build_raw(root, n=60):
    os.makedirs(f"{root}/arxiv_pdfs", exist_ok=True)
    for i in range(n):
        body = "Abstract\n\n1. Introduction\n" + wrap(words("research_paper", 500)) + "\nReferences\n[1] Smith et al."
        make_pdf(f"{root}/arxiv_pdfs/p{i}.pdf", body)
    os.makedirs(f"{root}/edgar", exist_ok=True)
    with open(f"{root}/edgar/test.jsonl", "w") as fh:
        for i in range(n):
            fh.write(json.dumps({"filename": f"{i}_10K.txt", "section_1": words("financial_filing", 400),
                                 "section_1A": words("financial_filing", 300), "section_3": words("legal_document", 250),
                                 "section_9": ""}) + "\n")
    os.makedirs(f"{root}/alaska_caselaw", exist_ok=True)
    for i in range(n):
        txt = "Opinion of the Court\nNo. S-1234\nSupreme Court of Alaska\n" + wrap(words("legal_document", 400))
        json.dump({"id": i, "casebody": {"opinions": [{"text": txt, "type": "majority"}]}},
                  open(f"{root}/alaska_caselaw/c{i}.json", "w"))
    os.makedirs(f"{root}/agnews", exist_ok=True)
    pd.DataFrame({"text": [words("news_article", 40) for _ in range(n)], "label": 2}).to_parquet(f"{root}/agnews/agnews.parquet")
    os.makedirs(f"{root}/gutenberg", exist_ok=True)
    gut = lambda body: ("The Project Gutenberg eBook of X\n*** START OF THE PROJECT GUTENBERG EBOOK X ***\n"  # noqa: E731
                        + wrap(body) + "\n*** END OF THE PROJECT GUTENBERG EBOOK X ***\nLicense text")
    pd.DataFrame({"text": [gut(words("book_literature", 600)) for _ in range(n)]}).to_parquet(f"{root}/gutenberg/en.parquet")
    rows = []
    for lg, base in BOOK_TEXT.items():
        for _ in range(n // 3):
            rows.append({"text": gut(base * 12 + words("book_literature", 50)), "language": lg})
    pd.DataFrame(rows).to_parquet(f"{root}/gutenberg/multi.parquet")
    os.makedirs(f"{root}/probe", exist_ok=True)
    rv = ["great product works well would buy again the battery lasts long and shipping was fast " * 2 for _ in range(30)]
    pd.DataFrame({"text": rv}).to_parquet(f"{root}/probe/amazon_reviews.parquet")
    nz = [" ".join("".join(rng.choice("abcdefghijklmnopqrstuvwxyz0123456789") for _ in range(rng.randint(2, 9))) for _ in range(60)) for _ in range(20)]
    pd.DataFrame({"text": nz}).to_parquet(f"{root}/probe/cc_noise.parquet")
    os.makedirs(f"{root}/multieurlex", exist_ok=True)
    for lg in ("de", "fr", "es", "it", "en"):
        recs = [{"celex_id": f"3{1990 + j}R{j:04d}", "text": {"header": "REGULATION No %d" % j, "recitals": LANG_TEXT[lg] * 3,
                                                               "main_body": [LANG_TEXT[lg] * 6, LANG_TEXT[lg] * 6]}} for j in range(20)]
        pd.DataFrame(recs).to_parquet(f"{root}/multieurlex/{lg}.parquet")


def run(args):
    print("\n$ " + " ".join(args))
    r = subprocess.run(args, cwd=ROOT, capture_output=True, text=True)
    print(r.stdout[-2500:])
    if r.returncode != 0:
        print(r.stderr[-3000:])
        sys.exit(f"FAILED: {' '.join(args)}")


def main():
    tmp = tempfile.mkdtemp(prefix="docclf_smoke_")
    raw, base = f"{tmp}/raw", f"{tmp}/data"
    build_raw(raw)
    cfg = yaml.safe_load(open(f"{ROOT}/configs/data.yaml"))
    cfg["paths"] = {"canonical_raw": f"{base}/raw_canonical", "processed": f"{base}/processed/docs",
                    "experiments": f"{base}/experiments", "gold": f"{base}/gold"}
    S = cfg["sources"]
    S["arxiv"]["path"] = f"{raw}/arxiv_pdfs"
    S["edgar"]["path"] = f"{raw}/edgar/test.jsonl"
    S["alaska_caselaw"]["path"] = f"{raw}/alaska_caselaw"
    S["agnews"]["path"] = f"{raw}/agnews/agnews.parquet"
    S["gutenberg_en"]["path"] = f"{raw}/gutenberg/en.parquet"
    S["gutenberg_multi"]["path"] = f"{raw}/gutenberg/multi.parquet"
    S["probe_reviews"]["path"] = f"{raw}/probe/amazon_reviews.parquet"
    S["probe_noise"]["path"] = f"{raw}/probe/cc_noise.parquet"
    S["multieurlex"]["files"] = {lg: f"{raw}/multieurlex/{lg}.parquet" for lg in ("de", "fr", "es", "it", "en")}
    # small synthetic corpus: shrink split counts so every class has data
    cfg["prepare"].update({"max_per_class": 60, "split_counts": {"train": 30, "val": 10, "test": 10}})
    cfg["prepare"]["min_tokens"] = {k: 20 for k in cfg["prepare"]["min_tokens"]}
    cfg["prepare"]["min_tokens"]["agnews"] = 10
    cfg["exp1"]["long_windows"] = [128, 256]
    cfg["exp2"].update({"gold_random_per_class": 5, "gold_hard_total": 10, "gold_hard_per_class_cap": 3,
                        "relabel_n": 8, "bias_items": 6})
    cfg["exp3"].update({"partB_per_class": 8, "chunk_chars": 800})
    cpath = f"{tmp}/data.yaml"
    yaml.safe_dump(cfg, open(cpath, "w"))

    run([PY, "src/adapters.py", "--config", cpath, "--inspect"])
    run([PY, "src/adapters.py", "--config", cpath])
    run([PY, "src/prepare_spark.py", "--config", cpath])
    B = [PY, "src/build_experiments.py"]
    run(B + ["exp1", "--config", cpath])
    run(B + ["probe-sample", "--config", cpath])
    run(B + ["leakage", "--config", cpath])
    run(B + ["gold-random", "--config", cpath])
    run([PY, "src/baselines.py", "--config", cpath, "--no-embeddings", "--boot", "50"])
    run(B + ["gold-hard", "--config", cpath, "--margins", f"{base}/experiments/margins.csv"])
    sheet = pd.read_csv(f"{base}/gold/gold_random_annotation.csv")
    key = pd.read_csv(f"{base}/gold/gold_random_key.csv").set_index("item_id")
    sheet["my_label"] = sheet.item_id.map(key.source_label)
    sheet.to_csv(f"{tmp}/gold_random_filled.csv", index=False)
    run(B + ["gold-relabel", "--config", cpath, "--labels", f"{tmp}/gold_random_filled.csv"])
    preds = f"{base}/experiments/preds.jsonl"
    run(B + ["placeholder-preds", "--config", cpath, "--out", preds, "--variants", "stripped_random", "noise10"])
    run(B + ["exp2-judge", "--config", cpath, "--preds", preds, "--key", f"{base}/gold/gold_random_key.csv",
             "--gold", f"{tmp}/gold_random_filled.csv"])
    run(B + ["placeholder-judge", "--config", cpath])
    run(B + ["exp3", "--config", cpath])
    print("\nSMOKE TEST PASSED. Temp dir:", tmp)


if __name__ == "__main__":
    main()
