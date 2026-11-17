
import hashlib
import random
import re
import unicodedata

CLASSES = ["research_paper", "financial_filing", "legal_document",
           "news_article", "book_literature"]


SOURCE_LABEL = {
    "arxiv": "research_paper",
    "edgar": "financial_filing",
    "alaska_caselaw": "legal_document",
    "agnews": "news_article",
    "gutenberg_en": "book_literature",
    "gutenberg_multi": "book_literature",
    "multieurlex": "legal_document",
    "probe_reviews": "other", "probe_noise": "other",
}

SOURCE_ROLE = {
    "arxiv": "main", "edgar": "main", "alaska_caselaw": "main",
    "agnews": "main", "gutenberg_en": "main",
    "gutenberg_multi": "ml_books", "multieurlex": "ml_legal",
    "probe_reviews": "probe", "probe_noise": "probe",     
}
CANON_COLS = ["doc_id", "source", "role", "label", "lang_declared", "text", "meta"]


# ----------------------------------------------------------------- utilities
def seeded_rng(*parts):
    """Deterministic RNG keyed by any parts (doc id, variant, window...)."""
    key = "|".join(str(p) for p in parts).encode("utf-8")
    return random.Random(int(hashlib.sha1(key).hexdigest()[:12], 16))


def collapse(text):
    return re.sub(r"\s+", " ", text or "").strip()


def n_tokens(text):
    return len((text or "").split())


def content_hash(text):
    return hashlib.sha1(collapse(text).lower()[:2000].encode("utf-8")).hexdigest()


# ------------------------------------------------- normalization
_LIGATURES = {"\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl",
              "\ufb03": "ffi", "\ufb04": "ffl", "\u00ad": ""}


def fix_artifacts(text):
    """Remove extraction artifacts so extraction style is not a class cue."""
    t = unicodedata.normalize("NFKC", text or "")
    for k, v in _LIGATURES.items():
        t = t.replace(k, v)
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"(\w)-\n(\w)", r"\1\2", t)                       # de-hyphenate
    t = re.sub(r"(?mi)^\s*(page\s+)?\d{1,4}(\s+of\s+\d{1,4})?\s*$", "", t)  # page numbers
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


_GUT_START = re.compile(r"\*\*\*\s*START OF (THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.I | re.S)
_GUT_END = re.compile(r"\*\*\*\s*END OF (THE|THIS) PROJECT GUTENBERG EBOOK.*", re.I | re.S)
_GUT_LINE = re.compile(r"^\s*(produced by|transcribed|transcriber|e-?text prepared|proofread)", re.I)


def strip_gutenberg(text):
    m = _GUT_START.search(text)
    if m:
        text = text[m.end():]
    text = _GUT_END.sub("", text)
    keep = [ln for ln in text.split("\n")
            if "gutenberg" not in ln.lower() and not _GUT_LINE.match(ln)]
    return "\n".join(keep)


_CASE_HEAD = [
    r"^\s*(opinion of the court|opinion|per curiam)\s*\.?\s*$",
    r"^\s*no\.\s*[a-z0-9\-]+.{0,60}$",
    r"^\s*(supreme court|court of appeals|superior court).{0,80}$",
    r"^\s*(argued|submitted|decided|filed|rehearing denied).{0,80}$",
    r"^\s*(appeal from|before:).{0,200}$",
    r"^\s*[A-Z][A-Za-z\.\, ]{2,60}\bv\.\s.{2,80}$",              # caption "A v. B"
]


def strip_caselaw_headmatter(text, max_lines=25):
    lines = text.split("\n")
    head, rest = lines[:max_lines], lines[max_lines:]
    head = [ln for ln in head if not any(re.match(p, ln, re.I) for p in _CASE_HEAD)]
    return "\n".join(head + rest)


def normalize(text, source):
    """Source-specific boilerplate removal + uniform artifact fixes.
    Keeps line structure (needed by generic_strip_structure and the
    leakage check)."""
    t = fix_artifacts(text)
    if source.startswith("gutenberg"):
        t = strip_gutenberg(t)
    if source == "alaska_caselaw":
        t = strip_caselaw_headmatter(t)
    return fix_artifacts(t)


def detect_lang(text):
    try:
        from langdetect import DetectorFactory, detect
        DetectorFactory.seed = 0
        c = collapse(text)
        return detect(c[:1500]) if len(c) >= 40 else "unk"
    except Exception:
        return "unk"


# ----------------------------------------------- freeform transform (Exp 1)
def generic_strip_structure(text):
    """Remove title/heading/boilerplate cues, then flatten to one line."""
    out = []
    lines = (text or "").split("\n")
    lens = sorted(len(x.strip()) for x in lines if x.strip())
    med = lens[len(lens) // 2] if lens else 0
    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        words = s.split()
        # heading-like: clearly shorter than the document's typical line, few
        # words, no sentence punctuation.  (Hard-wrapped lines are near full
        # width, so wrapped prose is not mistaken for headings.)
        if len(s) < 0.6 * med and len(words) <= 8 and not re.search(r"[.!?;,:]$", s):
            continue
        if s.isupper() and len(words) <= 15:
            continue
        if re.match(r"^(abstract|introduction|references|keywords|conclusions?)\b", s, re.I) and len(words) <= 4:
            continue
        if re.match(r"^\d+(\.\d+)*\.?\s+[A-Z]", s) and len(words) <= 10:
            continue                                              # numbered headings
        if re.match(r"^(item\s+\d+[a-c]?\.?|part\s+[ivx]+)\b", s, re.I) and len(words) <= 12:
            continue                                              # SEC item headings
        out.append(s)
    t = " ".join(out)
    t = re.sub(r"https?://\S+|\S+@\S+", " ", t)
    t = re.sub(r"\[\d+(?:[,\u2013-]\s*\d+)*\]", " ", t)            # [12], [3-5]
    return collapse(t)


def window_tokens(text, n, mode, rng):
    """Return an n-token window of text, or None if text is shorter than n."""
    toks = (text or "").split()
    if len(toks) < n:
        return None
    if mode == "first":
        w = toks[:n]
    elif mode == "random":
        s = rng.randint(0, len(toks) - n)
        w = toks[s:s + n]
    elif mode == "sme":                                           # start+middle+end
        k = n // 3
        mid = max(0, (len(toks) - k) // 2)
        w = toks[:k] + toks[mid:mid + k] + toks[-(n - 2 * k):]
    else:
        raise ValueError(f"unknown window mode {mode}")
    return " ".join(w)


def window_chars(text, n_chars, rng=None, fraction=None):
    """n_chars window at a random position, or at a fixed relative position
    (fraction in [0,1]) so parallel documents are cut at the same place."""
    t = text or ""
    if len(t) <= n_chars:
        return t
    room = len(t) - n_chars
    start = int(room * fraction) if fraction is not None else rng.randint(0, room)
    w = t[start:start + n_chars]
    if start > 0 and " " in w:
        w = w.split(" ", 1)[1]                                    # drop partial first word
    return w


_CONF = {"l": "1", "1": "l", "O": "0", "0": "O", "o": "0", "e": "c", "c": "e",
         "h": "b", "S": "5", "5": "S", "B": "8", "u": "v", "i": "l", "a": "o", "n": "u"}


def ocr_noise(text, rate, rng):
    """Character-level OCR-like noise: look-alike swaps, deletions, insertions."""
    out = []
    for ch in text:
        if ch.isspace() or rng.random() >= rate:
            out.append(ch)
            continue
        op = rng.random()
        if op < 0.5:
            out.append(_CONF.get(ch, rng.choice("abcdefghijklmnopqrstuvwxyz")))
        elif op < 0.7:
            continue
        elif op < 0.9:
            out.append(ch)
            out.append(rng.choice("abcdefghijklmnopqrstuvwxyz"))
        else:
            out.append(ch)
            out.append(ch)
    return "".join(out)


# ---------------------------------------------- leakage check features
def artifact_features(text):
    t = text or ""
    lines = [ln for ln in t.split("\n") if ln.strip()] or [""]
    lens = [len(ln) for ln in lines]
    n = max(len(t), 1)
    mean = sum(lens) / len(lens)
    std = (sum((x - mean) ** 2 for x in lens) / len(lens)) ** 0.5
    return {
        "n_chars": len(t), "n_lines": len(lines), "mean_line_len": mean, "std_line_len": std,
        "frac_nonascii": sum(ord(c) > 127 for c in t) / n,
        "frac_digit": sum(c.isdigit() for c in t) / n,
        "frac_upper": sum(c.isupper() for c in t) / n,
        "frac_punct": sum(c in ".,;:!?" for c in t) / n,
        "hyphen_eol": len(re.findall(r"-\n", t)) / len(lines),
    }
