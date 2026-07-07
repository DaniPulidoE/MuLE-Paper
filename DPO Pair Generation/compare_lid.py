"""Compare LID detectors for the windowed language-consistency labeler.

Runs langdetect, lingua (and fastText lid.176 if available) over French and
Portuguese candidates from iter1_questions.jsonl, using the SAME windowing and
decision logic as dataset_filter.py, and compares them against reference
buckets built from an English-stopword chunk heuristic:

  pure       -> 0 English chunks           (near-certain in-language answers)
  mostly_en  -> >=70% English chunks       (near-certain code-switched answers)
  random     -> unfiltered sample          (to size the ambiguous bucket)

Reported per detector and language:
  (a) yield loss on pure:        % of pure answers NOT labeled 1 (split into
                                 mislabeled-as-0 vs sent-to-ambiguous)
  (b) miss rate on mostly_en:    % of mostly-English answers NOT labeled 0
  (c) ambiguous rate on random:  % of random answers labeled -1
  (d) es-confusion on pure:      % of windows in pure answers classified 'es'
      (the pt row is the pt/es confusion that poisoned the rejected pool)
  plus wall-clock runtime.

Usage (from repo root):
  python "DPO Pair Generation/compare_lid.py" [--per_bucket 200] [--detectors langdetect,lingua,fasttext]
"""
import json
import re
import time
import random
import argparse
from collections import defaultdict

CANDIDATES_FILE = "./Datos/iter1_questions.jsonl"

# same constants as dataset_filter.py
WINDOW_SIZE = 350
WINDOW_STRIDE = 175
CONF_THRESHOLD = 0.8
INCONSISTENT_FRAC = 0.30
LEXICAL_EN_HITS = 3
CANDIDATE_LANGS = ("en", "fr", "pt", "es")
MAX_CHARS = 40000  # cap very long answers; windows beyond this add little signal

EN_EXCLUSIVE = re.compile(
    r"\b(the|which|wait|should|would|could|because|therefore|answer|actually|"
    r"let's|we're|doesn't|isn't|there|where|then|thus|hence|since|both|"
    r"again|another|something|anything)\b",
    re.IGNORECASE,
)


# copied from the MMATH eval - keep identical to dataset_filter._strip_math
def _strip_math(text):
    cleaned = re.sub(r'\$\$.*?\$\$|\\\(.*?\\\)|\\\[.*?\\\]', '', text, flags=re.DOTALL)
    cleaned = re.sub(r'\$[^$]*\$', '', cleaned)
    cleaned = re.sub(r'<[^>\n]{1,200}>', ' ', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned


def _windows(text):
    if len(text) <= WINDOW_SIZE:
        return [text]
    return [text[i:i + WINDOW_SIZE]
            for i in range(0, len(text) - WINDOW_SIZE + WINDOW_STRIDE, WINDOW_STRIDE)]


def _label_windows_with(classify, text, expected_lang):
    """Windowed three-way label for ONE section (no lexical check),
    mirroring dataset_filter._label_windows. Returns (label, window_labels)."""
    stripped = _strip_math(text)
    if len(stripped) < 40:
        return 1, []
    labels = [classify(w) for w in _windows(stripped)]
    n = len(labels)

    confirmed_off = 0
    i = 0
    while i < n:
        run_label = labels[i]
        j = i
        while j < n and labels[j] == run_label:
            j += 1
        if run_label is not None and run_label != expected_lang and (j - i >= 2 or n == 1):
            confirmed_off += j - i
        i = j

    n_target = sum(1 for lab in labels if lab == expected_lang)

    if confirmed_off / n >= INCONSISTENT_FRAC:
        return 0, labels
    if confirmed_off == 0 and n_target >= 0.5 * n:
        return 1, labels
    return -1, labels


def label_with(classify, text, expected_lang):
    """Per-section decision logic mirroring dataset_filter.language_consistency_label:
    split once at </think>, both sections must be consistent; lexical English
    check over the whole text. Also returns per-window labels (both sections)."""
    if "</think>" in text:
        think_pred, answer_pred = text.split("</think>", 1)
        lab_t, win_t = _label_windows_with(classify, think_pred, expected_lang)
        lab_a, win_a = _label_windows_with(classify, answer_pred, expected_lang)
        win_labels = win_t + win_a
        if 0 in (lab_t, lab_a):
            combined = 0
        elif -1 in (lab_t, lab_a):
            combined = -1
        else:
            combined = 1
    else:
        combined, win_labels = _label_windows_with(classify, text, expected_lang)

    if combined == 1 and expected_lang != "en":
        if len(EN_EXCLUSIVE.findall(_strip_math(text))) >= LEXICAL_EN_HITS:
            return -1, win_labels
    return combined, win_labels


# ---------------------------------------------------------------------------
# Reference buckets: English-stopword chunk heuristic (detector-independent)
# ---------------------------------------------------------------------------
STOP = {
    "en": {"the", "is", "are", "we", "so", "let", "wait", "which", "then", "of", "and",
           "to", "that", "this", "it", "be", "have", "can", "should", "would", "but",
           "if", "when", "need", "get", "was"},
    "fr": {"le", "la", "les", "des", "une", "est", "nous", "donc", "alors", "que",
           "qui", "pour", "dans", "avec", "cette", "sont", "mais", "si", "plus",
           "cela", "être", "fait", "tous", "comme", "aussi"},
    "pt": {"o", "os", "as", "um", "uma", "que", "para", "com", "não", "isso",
           "então", "vamos", "seja", "são", "mas", "se", "mais", "como", "temos",
           "ou", "por", "dos", "das", "ser", "está", "pelo", "essa"},
}
WORD = re.compile(r"[a-zA-Zàâáãçéèêëíîïóôõúûùüñ]+")
CHUNK_SPLIT = re.compile(r"(?<=[.!?;:])\s+|\n+")


def ref_bucket(text, target):
    """'pure' / 'mostly_en' / 'other' by English-chunk fraction."""
    chunks = [c for c in CHUNK_SPLIT.split(_strip_math(text)) if len(c.strip()) >= 40]
    if not chunks:
        return "other"
    en_set, tg_set = STOP["en"], STOP[target]
    n_en = 0
    for c in chunks:
        words = [w.lower() for w in WORD.findall(c)]
        e = sum(1 for w in words if w in en_set)
        t = sum(1 for w in words if w in tg_set)
        if e >= 2 and e > t:
            n_en += 1
    frac = n_en / len(chunks)
    if frac == 0:
        return "pure"
    if frac >= 0.7:
        return "mostly_en"
    return "other"


# ---------------------------------------------------------------------------
# Detector factories (each returns classify(window) -> code or None)
# ---------------------------------------------------------------------------
def make_langdetect():
    from langdetect import detect_langs, DetectorFactory
    DetectorFactory.seed = 42

    def classify(window):
        try:
            res = detect_langs(window)
        except Exception:
            return None
        cand = {r.lang: r.prob for r in res if r.lang in CANDIDATE_LANGS}
        total = sum(cand.values())
        if total <= 0:
            return None
        best = max(cand, key=cand.get)
        return best if cand[best] / total >= CONF_THRESHOLD else None
    return classify


def make_lingua():
    from lingua import Language, LanguageDetectorBuilder
    lingua_langs = {"en": Language.ENGLISH, "fr": Language.FRENCH,
                    "pt": Language.PORTUGUESE, "es": Language.SPANISH}
    code_of = {v: k for k, v in lingua_langs.items()}
    detector = (LanguageDetectorBuilder
                .from_languages(*lingua_langs.values())
                .with_preloaded_language_models()
                .build())

    def classify(window):
        values = detector.compute_language_confidence_values(window)
        if not values:
            return None
        top = values[0]
        return code_of[top.language] if top.value >= CONF_THRESHOLD else None
    return classify


def make_fasttext():
    import fasttext
    lid = fasttext.load_model("lid.176.bin")

    def classify(window):
        labels, probs = lid.predict(window, k=20)
        cand = {}
        for label, prob in zip(labels, probs):
            code = label.replace("__label__", "")
            if code in CANDIDATE_LANGS:
                cand[code] = prob
        total = sum(cand.values())
        if total <= 0:
            return None
        best = max(cand, key=cand.get)
        return best if cand[best] / total >= CONF_THRESHOLD else None
    return classify


DETECTOR_FACTORIES = {
    "langdetect": make_langdetect,
    "lingua": make_lingua,
    "fasttext": make_fasttext,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates_file", type=str, default=CANDIDATES_FILE)
    parser.add_argument("--per_bucket", type=int, default=200,
                        help="answers sampled per (language, bucket)")
    parser.add_argument("--detectors", type=str, default="langdetect,lingua,fasttext")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed)

    # ---- pass 1: stream file, reservoir-sample answers per (lang, bucket) ----
    reservoirs = {(l, b): [] for l in ("fr", "pt") for b in ("pure", "mostly_en", "random")}
    seen = {k: 0 for k in reservoirs}

    def offer(key, text):
        seen[key] += 1
        res = reservoirs[key]
        if len(res) < args.per_bucket:
            res.append(text)
        else:
            r = random.randint(0, seen[key] - 1)
            if r < args.per_bucket:
                res[r] = text

    print("Sampling answers (streaming the candidates file)...")
    n_rec = 0
    with open(args.candidates_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            lang = rec.get("language")
            if lang not in ("fr", "pt"):
                continue
            n_rec += 1
            if n_rec % 500 == 0:
                print(f"  {n_rec} fr/pt records...")
            for cand in rec["candidates"]:
                text = cand[:MAX_CHARS]
                offer((lang, "random"), text)
                bucket = ref_bucket(text, lang)
                if bucket in ("pure", "mostly_en"):
                    offer((lang, bucket), text)

    for (lang, bucket), res in sorted(reservoirs.items()):
        print(f"  {lang}/{bucket}: sampled {len(res)} of {seen[(lang, bucket)]} seen")

    # ---- pass 2: run each detector over the samples ----
    requested = [d.strip() for d in args.detectors.split(",") if d.strip()]
    results = {}
    for name in requested:
        try:
            classify = DETECTOR_FACTORIES[name]()
        except KeyError:
            print(f"\n[{name}] unknown detector, skipping")
            continue
        except Exception as e:
            print(f"\n[{name}] unavailable ({e}), skipping")
            continue

        print(f"\n[{name}] running...")
        t0 = time.time()
        stats = defaultdict(lambda: defaultdict(int))
        for lang in ("fr", "pt"):
            for bucket in ("pure", "mostly_en", "random"):
                for text in reservoirs[(lang, bucket)]:
                    lab, win_labels = label_with(classify, text, lang)
                    S = stats[(lang, bucket)]
                    S["n"] += 1
                    S[f"label_{lab}"] += 1
                    if bucket == "pure":
                        S["windows"] += len(win_labels)
                        S["es_windows"] += sum(1 for w in win_labels if w == "es")
                        S["off_windows"] += sum(
                            1 for w in win_labels if w is not None and w != lang)
        results[name] = (dict(stats), time.time() - t0)
        print(f"[{name}] done in {results[name][1]:.1f}s")

    # ---- report ----
    def pct(a, b):
        return f"{100 * a / b:5.1f}%" if b else "  n/a "

    print("\n" + "=" * 100)
    print(f"{'detector':<12}{'lang':<6}"
          f"{'pure->1':>9}{'pure->0':>9}{'pure->-1':>10}"
          f"{'mostlyEN->0':>13}{'random->-1':>12}"
          f"{'es-win|pure':>13}{'off-win|pure':>14}{'time':>8}")
    print("-" * 100)
    for name, (stats, dt) in results.items():
        for lang in ("fr", "pt"):
            P = stats.get((lang, "pure"), {})
            M = stats.get((lang, "mostly_en"), {})
            R = stats.get((lang, "random"), {})
            print(f"{name:<12}{lang:<6}"
                  f"{pct(P.get('label_1', 0), P.get('n', 0)):>9}"
                  f"{pct(P.get('label_0', 0), P.get('n', 0)):>9}"
                  f"{pct(P.get('label_-1', 0), P.get('n', 0)):>10}"
                  f"{pct(M.get('label_0', 0), M.get('n', 0)):>13}"
                  f"{pct(R.get('label_-1', 0), R.get('n', 0)):>12}"
                  f"{pct(P.get('es_windows', 0), P.get('windows', 0)):>13}"
                  f"{pct(P.get('off_windows', 0), P.get('windows', 0)):>14}"
                  f"{dt:>7.0f}s")
    print("=" * 100)
    print("Ideal detector: pure->1 near 100 (yield), pure->0 at 0 (mislabels), "
          "mostlyEN->0 near 100 (negatives caught),\nrandom->-1 small (ambiguous bucket), "
          "es-win|pure near 0 for pt (the pt/es confusion that poisoned rejected pools).")


if __name__ == "__main__":
    main()
