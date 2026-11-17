
import argparse
import glob
import json
import os
import sys

import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(__file__))
from docclf import CANON_COLS, SOURCE_LABEL, SOURCE_ROLE, seeded_rng  # noqa: E402


# ------------------------------------------------------------------ readers
def read_records(path):
    """Return a list of dict records from parquet / jsonl / json / csv / dir of json."""
    if os.path.isdir(path):
        recs = []
        for f in sorted(glob.glob(os.path.join(path, "**", "*.json"), recursive=True)):
            with open(f, encoding="utf-8") as fh:
                obj = json.load(fh)
            recs.extend(obj if isinstance(obj, list) else [dict(obj, _file=os.path.basename(f))])
        return recs
    ext = os.path.splitext(path)[1].lower()
    if ext == ".parquet":
        return pd.read_parquet(path).to_dict("records")
    if ext == ".jsonl":
        with open(path, encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]
    if ext == ".json":
        with open(path, encoding="utf-8") as fh:
            obj = json.load(fh)
        return obj if isinstance(obj, list) else [obj]
    if ext == ".csv":
        return pd.read_csv(path).to_dict("records")
    raise ValueError(f"unsupported file type: {path}")


def dig(obj, dotted):
    """Follow a dotted path through dicts; returns None if any step is missing."""
    cur = obj
    for part in dotted.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def coerce_text(value, lang=None):
    """Turn strings, lists of strings/dicts, or nested dicts into plain text."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        if lang and lang in value:
            return coerce_text(value[lang])
        if "text" in value:
            return coerce_text(value["text"])
        parts = [coerce_text(value[k]) for k in ("header", "recitals", "main_body") if k in value]
        return "\n".join(p for p in parts if p) if parts else "\n".join(coerce_text(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return "\n\n".join(coerce_text(v, lang) for v in value)
    return str(value)


_LANG_NAMES = {"english": "en", "german": "de", "french": "fr", "spanish": "es", "italian": "it",
               "portuguese": "pt", "dutch": "nl", "bulgarian": "bg", "greek": "el"}


def iso_lang(value):
    """'Spanish' -> 'es', 'ES' -> 'es', 'es-ES' -> 'es'; unknown names -> 'unk'."""
    v = str(value or "").strip().lower()
    if v in _LANG_NAMES:
        return _LANG_NAMES[v]
    return v[:2] if len(v) in (2, 5) and v[:2].isalpha() else "unk"


def row(source, doc_id, text, lang, meta=None):
    return {"doc_id": doc_id, "source": source, "role": SOURCE_ROLE[source],
            "label": SOURCE_LABEL[source], "lang_declared": lang,
            "text": text, "meta": json.dumps(meta or {}, ensure_ascii=False, default=str)}


def fail(source, msg, found):
    sys.exit(f"[{source}] {msg}\n  fields found: {sorted(found)[:40]}\n"
             f"  -> fix the field names for '{source}' in configs/data.yaml")


# ----------------------------------------------------------------- adapters
def load_pdf_dir(source, cfg, seed):
    from pypdf import PdfReader
    rows = []
    for i, f in enumerate(sorted(glob.glob(os.path.join(cfg["path"], "*.pdf")))):
        try:
            pages = PdfReader(f).pages[: cfg.get("max_pages", 8)]
            text = "\n".join((p.extract_text() or "") for p in pages)
        except Exception as e:                                   # corrupt PDF: skip, keep going
            print(f"  skip {os.path.basename(f)}: {e}")
            continue
        rows.append(row(source, f"{source}-{i:05d}", text, "en", {"file": os.path.basename(f)}))
    return rows


def load_edgar(source, cfg, seed):
    recs, rows = read_records(cfg["path"]), []
    pre = cfg.get("section_prefix", "section_")
    for i, r in enumerate(recs):
        secs = {k: v for k, v in r.items()
                if k.startswith(pre) and isinstance(v, str) and len(v.split()) >= cfg.get("min_section_tokens", 200)}
        if not secs:
            continue
        key = seeded_rng(seed, source, i).choice(sorted(secs))   # one random section per filing
        rows.append(row(source, f"{source}-{i:05d}", secs[key], "en",
                        {"section": key, "filing": r.get(cfg.get("id_field", "filename"))}))
    if not rows:
        fail(source, "no usable section text found", recs[0].keys() if recs else [])
    return rows


def load_caselaw(source, cfg, seed):
    recs, rows = read_records(cfg["path"]), []
    for i, r in enumerate(recs):
        text = ""
        for p in cfg["text_paths"]:
            text = coerce_text(dig(r, p))
            if text.strip():
                break
        if not text.strip():
            fail(source, "no text found with any of text_paths", r.keys())
        rows.append(row(source, f"{source}-{i:05d}", text, "en", {"file": r.get("_file")}))
    return rows


def load_table(source, cfg, seed):
    recs, rows = read_records(cfg["path"]), []
    if recs and cfg["text_field"] not in recs[0]:
        fail(source, f"text_field '{cfg['text_field']}' not found", recs[0].keys())
    for i, r in enumerate(recs):
        lang = (r.get(cfg["lang_field"]) if cfg.get("lang_field") else None) or cfg.get("lang_default") \
            or ("en" if source != "gutenberg_multi" else "unk")
        rows.append(row(source, f"{source}-{i:05d}", coerce_text(r[cfg["text_field"]]), iso_lang(lang),
                        {**{k: r[k] for k in ("title", "author", "id") if k in r}, "subtype": cfg.get("subtype")}))
    return rows


def load_multieurlex(source, cfg, seed):
    rows = []
    for lang, path in cfg["files"].items():
        if not os.path.exists(path):
            print(f"  [multieurlex] missing file for '{lang}': {path} (skipped)")
            continue
        for r in read_records(path):
            if cfg["celex_field"] not in r or cfg["text_field"] not in r:
                fail(source, "celex_field/text_field not found", r.keys())
            celex = str(r[cfg["celex_field"]])
            rows.append(row(source, f"{source}-{celex}-{lang}", coerce_text(r[cfg["text_field"]], lang),
                            lang, {"celex_id": celex}))
    return rows


LOADERS = {"pdf_dir": load_pdf_dir, "jsonl": None, "json_dir": None, "parquet": load_table}
SOURCE_LOADER = {"arxiv": load_pdf_dir, "edgar": load_edgar, "alaska_caselaw": load_caselaw,
                 "agnews": load_table, "gutenberg_en": load_table, "gutenberg_multi": load_table,
                 "multieurlex": load_multieurlex,
                 "probe_reviews": load_table, "probe_noise": load_table}


def inspect_sources(cfg):
    for name, scfg in cfg["sources"].items():
        print(f"\n== {name}")
        try:
            if name == "arxiv":
                n = len(glob.glob(os.path.join(scfg["path"], "*.pdf")))
                print(f"  {n} PDFs in {scfg['path']}")
            elif name == "multieurlex":
                for lang, p in scfg["files"].items():
                    if os.path.exists(p):
                        recs = read_records(p)
                        print(f"  {lang}: {len(recs)} records; keys: {sorted(recs[0].keys())}")
                    else:
                        print(f"  {lang}: MISSING {p}")
            else:
                recs = read_records(scfg["path"])
                print(f"  {len(recs)} records; keys: {sorted(recs[0].keys())}")
        except Exception as e:
            print(f"  could not read: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--only", nargs="*", help="restrict to these source names")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    if args.inspect:
        return inspect_sources(cfg)
    rows = []
    for name, scfg in cfg["sources"].items():
        if args.only and name not in args.only:
            continue
        print(f"loading {name} ...")
        got = SOURCE_LOADER[name](name, scfg, cfg["seed"])
        print(f"  {len(got)} documents")
        rows.extend(got)
    df = pd.DataFrame(rows, columns=CANON_COLS)
    out = cfg["paths"]["canonical_raw"]
    os.makedirs(out, exist_ok=True)
    df.to_parquet(os.path.join(out, "docs.parquet"), index=False)
    print(f"wrote {len(df)} rows -> {out}/docs.parquet")
    print(df.groupby(["source", "lang_declared"]).size())


if __name__ == "__main__":
    main()
