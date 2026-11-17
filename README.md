# Document-type classification with LLMs: freeform text, LLM judges and other languages

A small, fully reproducible study of three practical questions in document classification:

1. **Freeform documents.** How well can a model classify a document with no reliable structure (no title or headings, a random excerpt, OCR errors)?
2. **LLM as judge.** Can a second LLM catch the first one's labeling mistakes?
3. **Other languages.** Can a model trained or prompted in English classify documents in other languages?


## Results at a glance

| Question | Finding | Key numbers |
|---|---|---|
| Q1 freeform | Structure barely matters to an LLM; the amount of text does. TF-IDF breaks on noise, the LLM does not. | 32-word stripped fragments: LLM 0.949 vs TF-IDF 0.812 macro-F1. At 10% character noise: 0.951 vs 0.698. 128+ words: 0.97 to 1.00. |
| Q1 "other" | The LLM can abstain directly; a classic model cannot. | Out-of-scope items called "other": LLM 72.7% vs 8.8% for TF-IDF with a confidence threshold. False "other" on real documents: 2.0% vs 4.1%. |
| Q2 judge | Judges catch random errors but not the worker's real ones: they share the worker's reading. Most "errors" matched the true source, which exposed bias in my hand labels. | Planted errors caught: 93%. Real disagreements caught: 2% (same-family judge), 7% (other-family judge). Worker matched the source on 35 of 42 disputed items. |
| Q3 languages | Language gap and domain gap are separate problems. A multilingual encoder fixes the first; only broader class definitions fix the second. | Same 70 EU laws: TF-IDF 0.36 in English, 0.00 elsewhere. Multilingual embeddings flat across 5 languages but low. LLM 1.00 in all 5. Legal vs books (fr/es): LLM 0.97. |

Total API cost of the study: **$0.49**. Measured cost at scale: about **$120 to $150 per million documents**.

Full write-up with methods, tables and limitations: [docs/results.md](docs/results.md).

## Data

Five classes, one public source each, plus a test-only out-of-scope set.

| Class | Source | Documents |
|---|---|---|
| research_paper | arXiv | 920 |
| financial_filing | EDGAR-CORPUS (2020) | 685 |
| legal_document | Caselaw Access Project, Alaska; MultiEURLEX (Exp 3) | 250; 70 laws x 5 languages |
| news_article | AG News | 300 |
| book_literature | Project Gutenberg (English; French and Spanish for Exp 3) | 300; 300 |
| other (test only) | Amazon reviews; Common Crawl pages | 300 |

Data is not included in this repository. Download each source into `data/raw/` (paths and field names are in `configs/data.yaml`).

## Method

- **Pipeline:** source adapters write one canonical Parquet table; a PySpark job normalizes text, removes extraction artifacts and boilerplate, deduplicates, detects language and splits 150/50/50 per class.
- **Experiment 1 (freeform):** each test document becomes several variants: opening with structure, random chunk with headings stripped, start/middle/end sample, and 2/5/10% OCR-like character noise, at 32 to 1024 words.
- **Experiment 2 (judge):** 150 hand-labeled gold items (100 random, 50 hardest for the baseline). Two judges, one from the worker's model family and one from another, check real worker labels and a copy with 20% of labels deliberately corrupted. Bias tests shuffle the class order or remove the rubric.
- **Experiment 3 (languages):** the same 70 laws in English, German, French, Spanish and Italian (any gap is caused by language alone), plus legal vs literature in French and Spanish.
- **Models:** TF-IDF + logistic regression; English and multilingual sentence embeddings; `gpt-6-luna` as the zero-shot LLM worker; `gemini-3.1-flash-lite` and `gpt-6-luna` as judges.
- **Prompt:** one English prompt for every language, built from `configs/taxonomy.yaml`; fixed instructions first and the document last so a provider can cache the shared prefix; JSON output validated with Pydantic.
- **Engineering:** every LLM call is cached and resumable, rate limits (requests per minute and per day, tokens per minute) are enforced in code, identical requests are sent once, and `--dry-run` prints call counts and cost before spending anything.
- **Statistics:** macro-F1 with 95% bootstrap confidence intervals.

## Run it

Requires Python 3.12, Java 17+ (for PySpark) and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
export OPENAI_API_KEY=...   
export GEMINI_API_KEY=...

python scripts/smoke_test.py                     # synthetic end-to-end check
python src/adapters.py                           # raw sources -> canonical table
python src/prepare_spark.py                      # normalize, dedup, language ID, split
python src/build_experiments.py exp1             # freeform variants
python src/build_experiments.py exp3             # multilingual sets
python src/baselines.py                          # TF-IDF + embedding baselines
python src/exp3_baselines.py                     # Exp 3 classic methods
python src/llm_experiments.py exp1  --model c    # LLM on freeform variants
python src/llm_experiments.py probe --model c    # out-of-scope probe
python src/llm_experiments.py exp3  --model c    # LLM on other languages
```

Experiment 2 (LLM as judge) needs a hand-labeled gold set:

```bash
python src/build_experiments.py gold-random        # 100 items to label (labels hidden)
python src/build_experiments.py gold-hard --margins data/experiments/margins.csv   # 50 hardest items
# fill my_label in both sheets, then merge them into data/gold/gold_all_filled.csv and gold_all_key.csv
python src/llm_runner.py worker --model c --gold data/gold/gold_all_filled.csv
python src/build_experiments.py exp2-judge --preds data/experiments/preds_gpt-6-luna.jsonl \
    --key data/gold/gold_all_key.csv --gold data/gold/gold_all_filled.csv
python src/llm_runner.py judge --model c           # same-family judge
python src/llm_runner.py judge --model b           # different-family judge
python src/analyze_judge.py --key data/gold/gold_all_key.csv --gold data/gold/gold_all_filled.csv
```

Add `--dry-run` to any LLM command to see the number of calls and the cost first.

## Limitations

- Public sources: the LLM may have seen some documents in training, which could inflate its scores.
- Small samples (30 documents per class in Exp 1, 70 laws in Exp 3): small differences are not meaningful.
- One annotator for the gold set (kappa 0.659 against the source labels).



## Repository

```
configs/   data.yaml (paths, splits), llm.yaml (models, rate limits), taxonomy.yaml (class definitions)
src/       adapters, Spark preparation, experiment builders, baselines, LLM runner and experiments
scripts/   smoke test
results/   summary tables (no document text)
docs/      results write-up
```
