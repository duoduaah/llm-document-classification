import math
import re
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from docclf import CLASSES  # noqa: E402

LEXICON = {
    "research_paper": ["we propose", "abstract", "experiments", "dataset", "et al", "theorem",
                       "results show", "algorithm", "proof", "baseline", "we show", "model"],
    "financial_filing": ["revenue", "fiscal", "net income", "shareholders", "risk factors", "company",
                         "operating", "securities", "stock", "cash flows", "subsidiaries"],
    "legal_document": ["court", "plaintiff", "defendant", "shall", "pursuant", "appeal", "statute",
                       "article", "hereby", "judgment", "regulation", "commission"],
    "news_article": ["said", "reuters", "percent", "on tuesday", "announced", "officials",
                     "yesterday", "reported", "ap", "monday"],
    "book_literature": ["he said", "she said", "her", "his", "once", "chapter", "eyes", "heart",
                        "night", "the old", "had been"],
}


def _scores(text):
    t = " " + text.lower() + " "
    return {c: sum(len(re.findall(r"\b" + re.escape(k) + r"\b", t)) for k in kws)
            for c, kws in LEXICON.items()}


class PlaceholderWorker:
    name = "placeholder-worker"

    def predict(self, texts):
        out = []
        for t in texts:
            sc = _scores(t)
            ranked = sorted(sc.items(), key=lambda kv: -kv[1])
            (best, b), (second, s) = ranked[0], ranked[1]
            if b == 0:
                out.append({"label": "other", "confidence": 0.3, "evidence": "no keyword matched",
                            "ambiguous": True, "runner_up": None})
                continue
            conf = 1 / (1 + math.exp(-(b - s)))
            out.append({"label": best, "confidence": round(conf, 3),
                        "evidence": f"placeholder keyword score {b} vs {s}",
                        "ambiguous": (b - s) <= 1, "runner_up": second if s > 0 else None})
        return out


class PlaceholderJudge:
    name = "placeholder-judge"

    def verify(self, texts, proposed_labels):
        out = []
        for t, lab in zip(texts, proposed_labels):
            sc = _scores(t)
            top = max(sc, key=sc.get)
            if sc[top] == 0:
                out.append({"verdict": "unsure", "suggested_label": None, "reason": "no keywords"})
            elif lab == top or sc.get(lab, 0) >= sc[top]:
                out.append({"verdict": "correct", "suggested_label": None, "reason": "label supported"})
            else:
                out.append({"verdict": "incorrect", "suggested_label": top, "reason": f"{top} scores higher"})
        return out


def get_worker(provider="placeholder", **kw):
    if provider == "placeholder":
        return PlaceholderWorker()
    if provider == "gemini":
        from llm_runner import GeminiWorker
        return GeminiWorker(kw.get("config", "configs/llm.yaml"), kw.get("model", "a"))
    raise NotImplementedError(f"unknown provider {provider}")


def get_judge(provider="placeholder", **kw):
    if provider == "placeholder":
        return PlaceholderJudge()
    if provider == "gemini":
        from llm_runner import GeminiJudge
        return GeminiJudge(kw.get("config", "configs/llm.yaml"), kw.get("model", "b"))
    raise NotImplementedError(f"unknown provider {provider}")
