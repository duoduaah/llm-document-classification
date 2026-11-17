# Results

A zero-shot LLM classified freeform, noisy and non-English documents at 94 to 100%, but an LLM judge could not catch the worker's real mistakes. Total API spend: $0.49.

| Question | Finding | Headline number |
| --- | --- | --- |
| Q1. How do you classify freeform documents? | Structure barely matters to an LLM; the amount of text does. TF-IDF breaks on noise, the LLM does not. | 32-word stripped fragments: LLM 0.949 vs TF-IDF 0.812 macro-F1; at 10% OCR noise 0.951 vs 0.698 |
| Q2. Can an LLM act as a judge? | It catches random errors but not the worker's real ones, because it shares the worker's reading. Most of those "errors" matched the source label, which exposed bias in the hand labels. | Judge caught 93% of planted errors, 2 to 7% of real ones |
| Q3. Can an English-trained model handle other languages? | A word model fails outright; a multilingual encoder removes the language gap but not a domain gap; an LLM with English instructions handles all five languages. | Same 70 laws: TF-IDF 0.36 English, 0.00 elsewhere; LLM 1.00 in all five |

## Setup

Five document types, all from public data, processed with a PySpark pipeline into one table and evaluated against hidden source labels.

| Class | Source | Documents |
| --- | --- | --- |
| research_paper | arXiv PDFs | 920 collected |
| financial_filing | EDGAR-CORPUS 2020 | 685 |
| legal_document | Alaska case law (+ MultiEURLEX legislation for Exp 3) | 250 (+ 70 laws x 5 languages) |
| news_article | AG News | 300 |
| book_literature | Project Gutenberg, English (+ French, Spanish for Exp 3) | 300 (+ 300) |
| other (test only) | Amazon reviews + Common Crawl pages | 300 |

- **Split:** 150 train / 50 val / 50 test per class, deduplicated before splitting.
- **Gold set:** 150 hand-labeled items (100 random + 50 hardest for the baseline). Agreement with the source label 0.727, Cohen's kappa 0.659; 21% marked ambiguous.
- **Baseline:** TF-IDF + logistic regression trained on the train split; sentence-embedding baselines for Exp 3.
- **LLM worker:** gpt-6-luna, zero-shot, one English prompt built from `configs/taxonomy.yaml`, JSON output validated with Pydantic.
- **LLM judges:** gemini-3.1-flash-lite (different family) and gpt-6-luna (same family as the worker).
- **Runs:** every call cached, rate-limited and resumable; 30 test documents per class for Exp 1, all items for the probe and Exp 3.
- **Main metric:** macro-F1 (F1 averaged over classes, so each class counts equally), with 95% bootstrap confidence intervals.

## Q1. Freeform documents (Experiment 1)

Removing titles, headings and the document opening cost the LLM 2.4 points, within noise; OCR noise cost it nothing. TF-IDF lost 11 points to noise.

"Freeform" here means a document with no reliable structure: a random excerpt, headings stripped, OCR-like character errors.

### 32-word fragments, macro-F1

| Text the model saw | TF-IDF | LLM (zero-shot) | LLM 95% CI |
| --- | --- | --- | --- |
| Document opening, headings kept | 0.864 | 0.973 | 0.944 to 0.994 |
| Random chunk, headings removed | 0.812 | 0.949 | 0.913 to 0.976 |
| + 2% character noise | 0.796 | 0.949 | 0.913 to 0.975 |
| + 5% character noise | 0.777 | 0.937 | 0.898 to 0.966 |
| + 10% character noise | 0.698 | 0.951 | 0.917 to 0.980 |

TF-IDF was trained on stripped text and scored on the full test split; the LLM was scored on 30 documents per class. The gaps are far wider than either interval.

### Longer text

- At 128 words or more, every LLM variant scored 0.97 to 1.00, structured or stripped, clean or noisy.
- At 256 words, sampling start + middle + end (0.981) did no better than one random chunk (0.995).
- The 1024-word row covers only the 64 longest documents, so its 1.000 is not representative.

### Where it fails

Books are the weak class: 25 of 30 short book fragments correct (2 called research papers, 2 "other", 1 legal). Every other class scored 27 to 30 of 30. Books were also the weakest class for the human labeler (agreement 0.59) and for the baseline: Gutenberg mixes novels with essays and treatises.

### Saying "other" (out-of-scope probe)

| Method | Says "other" on probe items | Wrongly says "other" on real documents |
| --- | --- | --- |
| TF-IDF + confidence threshold | 8.8% | 4.1% |
| LLM, allowed to answer "other" | 72.7% | 2.0% |

- Reviews: the LLM said "other" on 93% (n=123). Web pages: 43% (n=82); the rest went mostly to books (30) and news (12), because web prose resembles them.
- TF-IDF has no "none of these" option; its confidence separates junk from real documents only weakly (AUROC 0.728).
- 205 of 300 probe items were long enough for a 32-word window.

## Q2. LLM as judge (Experiment 2)

Both judges caught 93% of planted errors but only 2 to 7% of the worker's real disagreements with the hand labels, because they agree with the worker, and the worker usually agreed with the source.

**Setup.** The worker labeled the 150 gold items. Each judge then saw a document plus a proposed label and answered correct, incorrect or unsure. Two conditions: the worker's real labels, and a synthetic copy where 20% of labels were swapped to a random wrong class.

### Can the judge catch wrong labels? (truth = hand labels)

| Judge | Condition | Wrong labels | Caught (recall) | Precision | False alarms on correct labels |
| --- | --- | --- | --- | --- | --- |
| gpt-6-luna (same family) | real | 42 | 0.02 | 1.00 | 0 |
| gemini-3.1-flash-lite (different family) | real | 42 | 0.07 | 0.75 | 1 |
| gpt-6-luna | synthetic | 30 | 0.93 | 0.45 | 34 |
| gemini-3.1-flash-lite | synthetic | 30 | 0.93 | 0.44 | 35 |

### Why: the judges side with the worker

- Worker accuracy against the hand labels was 72% (108 of 150); it was 80% on a 40-item pilot with no hard items.
- All 34 of the same-family judge's false alarms, and 34 of the other judge's 35, fell on items where the worker disagreed with the hand label. The same-family judge suggested the worker's label every time; the other judge did 30 times.
- On 35 of the 42 disagreements, the worker matched the document's real source label and the hand label did not.
- 18 of the 42 (43%) had been marked ambiguous by the labeler, against 13% on items where worker and labeler agreed.
- 24 of the 42 are fragments hand-labeled as news; 11 of those came from court opinions, which read like crime reporting in 32 words.

So most "worker errors" were not errors about what the document is. The judge was not blind; the gold labels had a systematic lean towards news on short fragments, found by analyzing disagreements. Planted errors are random and obvious; real disagreements are plausible, and a second LLM shares them.

### Bias tests

Shuffling the order of the classes, or removing all definitions and rules, changed 0% of verdicts for both judges (100 items each). The judges rely on their own idea of each class, not on the rubric.

### Same family vs different family

The same-family judge caught 1 real error, the other caught 3. That is too few to show self-preference either way.

## Q3. Other languages (Experiment 3)

The language gap and the domain gap are separate problems: a multilingual encoder fixed the first; only the LLM's broader class definitions fixed the second.

**Part A** uses the same 70 EU laws in English, German, French, Spanish and Italian, so any difference between languages is caused by language alone. Every item is a law; the score is the share labeled legal_document (legal recall). Classic models were trained on English data only.

### Part A: legal recall on the same 70 laws

| Method | Condition | en | de | fr | es | it |
| --- | --- | --- | --- | --- | --- | --- |
| TF-IDF | start | 0.36 | 0.00 | 0.00 | 0.00 | 0.00 |
| TF-IDF | freeform | 0.36 | 0.00 | 0.00 | 0.00 | 0.00 |
| English embeddings | start | 0.09 | 0.00 | 0.56 | 0.03 | 0.07 |
| Multilingual embeddings | start | 0.39 | 0.39 | 0.37 | 0.41 | 0.33 |
| Multilingual embeddings | freeform | 0.24 | 0.31 | 0.24 | 0.27 | 0.24 |
| LLM (English prompt) | start and freeform | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

"start" is the first 2,000 characters; "freeform" is a stripped chunk from 30% into the document. With 70 laws per cell, differences under about 0.10 are noise.

### What it shows

1. **TF-IDF collapses outside English.** It learned English words; it has no way to read "Gericht" or "doit".
2. **English embeddings are noise.** French at 0.56 beats English: an English encoder scrambles French into vectors that happen to land near "legal". Right for the wrong reason.
3. **Multilingual embeddings remove the language gap.** German, French, Spanish and Italian match English. But all are low, English included.
4. **The low English score is a domain gap.** The classic models learned "legal" only from court opinions; these are statutes. The LLM's definition names legislation, and it scored 1.00 everywhere.

### Part B: legal vs books, accuracy

| Method | French | Spanish |
| --- | --- | --- |
| TF-IDF | 0.00 | 0.00 |
| English embeddings | 0.61 | 0.55 |
| Multilingual embeddings | 0.65 | 0.65 |
| LLM | 0.97 | 0.97 |

Part B matters because Part A has one class: a model always answering "legal" would also score 1.00 there. Italian is excluded because no Italian books were collected. The LLM hit the ceiling in Part A, so this test cannot rank it against a stronger LLM.

## Cost and scale

The whole study cost $0.49 in API fees; at measured usage, classifying one million documents would cost about $120 to $150.

| Model | Role | Calls | Input tokens | Output tokens | Cost |
| --- | --- | --- | --- | --- | --- |
| gpt-6-luna | worker | 2,920 | 2,934,847 | 304,370 | $0.45 |
| gpt-6-luna | judge | 352 | 229,934 | 34,383 | $0.04 |
| gemini-3.1-flash-lite | judge | 352 | 238,038 | 16,137 | free tier |
| gemini-3.1-flash-lite | pilot | 41 | 30,895 | 2,890 | free tier |

Prices: gpt-6-luna $0.10 input and $0.50 output per million tokens ([OpenAI pricing](https://developers.openai.com/api/docs/pricing.md), checked 2026-11-10).

- **Per document:** about 725 input + 90 output tokens on a 32-word fragment ($118 per million documents); about 1,005 + 104 averaged over all runs ($152 per million).
- **No labeled data needed:** the LLM is zero-shot. TF-IDF is free to run but needed 150 labeled documents per class and lost 14 to 25 points on freeform text.
- **Send excerpts, not documents:** accuracy reached 0.97+ at 128 words, so cost per document stays fixed however long the document is.
- **Caching:** the fixed instructions (about 720 tokens) come first in every prompt so a provider can cache them. No caching occurred: the prompt was below the caching minimum. At production scale with a longer taxonomy, that ordering is what cuts input cost.
- **Deduplication:** identical requests were sent once; the 500 judge requests became 352 calls per judge.
- **Free-tier limits shaped the design:** gemini-2.5-flash-lite allowed 20 requests a day and was dropped; the free Gemini model got the role with the fewest calls.

## Limitations

- **Possible contamination:** arXiv, Gutenberg, EDGAR and EU legislation are public, so the LLM may have seen some documents in training. This could inflate scores.
- **Small samples:** 30 documents per class in Exp 1 and 70 laws in Exp 3 give intervals of several points; small differences are not meaningful.
- **One annotator:** the gold set has a single labeler with kappa 0.659 against the source, and a measured lean towards news on short fragments.
- **Five classes only:** a real document platform has many more types, with closer neighbours (slides, manuals, worksheets) that would be harder.
- **Ceiling effects:** the LLM scored 1.00 in Exp 3 Part A, so that test cannot separate stronger models.
- **One source per class:** some of what a model learns may be corpus style rather than document type.
- **Leakage check:** a model using only formatting features scored 0.528 on stripped text across 5 classes, so some formatting signal survives stripping; it was not fully explained.

## Future work

- Blind adjudication of the 42 disputed gold items, then re-score both judges against adjudicated labels.
- Translate-then-classify for Exp 3, and a harder legal set (contracts) that avoids the ceiling.
- A second annotator on a subset to measure human agreement directly.
