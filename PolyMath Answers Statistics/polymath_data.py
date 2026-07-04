import json
import os
import re
import sys

# rewards.py (acc_compute_score) lives in "DPO Pair Generation", a sibling folder.
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "DPO Pair Generation"))
from rewards import acc_compute_score

try:
    from langdetect import detect_langs, DetectorFactory, LangDetectException
    DetectorFactory.seed = 42
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False

MODEL_NAME = "XueZhang-bjtu/1.5B-cold-start-SFT"
RESULTS_DIR = "../evaluation/logs-eval/PolyMath-temp_0.9/1.5B-cold-start-SFT"
LEVELS = ["low", "medium", "high", "top"]
LANGS = ["en", "es", "fr", "pt"]
NUM_SAMPLES = 4

# Extends the backtrack-signal regex from "RL Train Answers Statistics" with more phrases per
# language (incl. Spanish, previously missing) so detection density is comparable across all
# four scoped languages (en/es/fr/pt) rather than skewed toward whichever has more patterns.
BACKTRACK_SIGNALS = re.compile(
    r'\bwait\b|\bactually\b|\bno,?\s+wait\b|'
    r'\blet me reconsider\b|\blet me restart\b|'
    r"that'?s wrong\b|\bi made an error\b|"
    r'\blet me try again\b|\bhmm,?\s+actually\b|'
    r'\bhold on\b|\bon second thought\b|'
    r'\bscratch that\b|\blet me redo (?:this|that)\b|'
    r'\blet me recheck\b|\blet me double.?check\b|'
    r"that(?:'s| is) (?:not right|incorrect)\b|"
    r'\bi need to reconsider\b|\boops\b|\bmy mistake\b|'
    r'\blet me start over\b|\bupon further (?:thought|reflection)\b|'
    # French
    r'\battends\b|\ben fait\b|\bnon,?\s+attendez\b|'
    r'\bje me suis tromp|\breprenons\b|'
    r'\boubl(?:ie|iez) ça\b|\blaisse-moi refaire\b|'
    r"(?:ce n'est pas correct|c'est faux)\b|\bmon erreur\b|"
    # Portuguese
    r'\bespera\b|\bna verdade\b|\bnão,?\s+espera\b|'
    r'\bcometi um erro\b|\besquece isso\b|\bdeixa eu refazer\b|'
    r'\bisso está (?:errado|incorreto)\b|\bmeu erro\b|\bpensando bem\b|'
    # Spanish
    r'\bespera\b|\ben realidad\b|\bno,?\s+espera\b|'
    r'\bcometí un error\b|\beso (?:está mal|es incorrecto)\b|'
    r'\bun momento\b|\bpensándolo bien\b|\bdéjame reconsiderar\b',
    re.IGNORECASE
)


def format_reward(text):
    """Checks for <think>/</think> tags and a \\boxed{} answer shortly after </think>."""
    if not re.search(r'<think>', text):
        return 0
    if not re.search(r'</think>', text):
        return 0

    close_pos = text.rfind('</think>')
    boxes_after = [m.start() for m in re.finditer(r'\\boxed\{', text)
                   if m.start() > close_pos]
    if not boxes_after:
        return 0
    if (min(boxes_after) - close_pos) > 800:
        return 0
    return 1


def language_consistency_score(text, expected_lang):
    """Returns a continuous language consistency score in [0.0, 1.0]."""
    if not LANGDETECT_AVAILABLE:
        return 1.0

    text_for_detection = re.sub(r'\\boxed\{[^}]*\}', '', text)
    text_for_detection = re.sub(r'\$[^$]*\$', '', text_for_detection)
    text_for_detection = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', '', text_for_detection)
    text_for_detection = re.sub(r'<think>|</think>', '', text_for_detection)
    text_for_detection = text_for_detection.strip()

    if len(text_for_detection) < 20:
        return 1.0

    try:
        lang_probs = detect_langs(text_for_detection)
        for lp in lang_probs:
            if lp.lang == expected_lang:
                return float(lp.prob)
        return 0.0
    except LangDetectException:
        return 0.0


def load_records(results_dir=None, levels=None, langs=None, num_samples=None, tokenizer=None):
    """Loads PolyMath generation JSONs (polymath_res_gen.py output) into per-sample records.

    Expects files at f"{results_dir}/{level}/{lang}.json", each a list of items with
    "answer" (ground truth) and "thinking_pred_i"/"answer_pred_i" for i in range(num_samples).

    Returns a list of dicts: Level, Language, Question ID, Accuracy, LC Score, Backtracks,
    Length, Format, Question Correct Count (samples correct out of num_samples for that question).
    """
    # Resolved lazily (rather than as default-arg values) so overriding the module-level
    # constants above after import still takes effect.
    if results_dir is None:
        results_dir = RESULTS_DIR
    if levels is None:
        levels = LEVELS
    if langs is None:
        langs = LANGS
    if num_samples is None:
        num_samples = NUM_SAMPLES

    records = []
    for level in levels:
        for lang in langs:
            path = os.path.join(results_dir, level, f"{lang}.json")
            if not os.path.exists(path):
                print(f"Warning: missing {path}, skipping.")
                continue
            with open(path, "r", encoding="utf-8") as f:
                items = json.load(f)

            for item in items:
                ground_truth = item.get("answer")
                if ground_truth is None:
                    continue

                samples = []
                for i in range(num_samples):
                    answer_pred = item.get(f"answer_pred_{i}")
                    if answer_pred is None:
                        continue
                    thinking_pred = item.get(f"thinking_pred_{i}") or ""
                    full_text = f"<think>{thinking_pred}</think>{answer_pred}"
                    acc = acc_compute_score(answer_pred, ground_truth)
                    samples.append((full_text, acc))

                if not samples:
                    continue

                correct_count = sum(1 for _, acc in samples if acc == 1.0)

                for full_text, acc in samples:
                    records.append({
                        "Level": level,
                        "Language": lang,
                        "Question ID": str(item.get("id", "")),
                        "Accuracy": acc,
                        "LC Score": language_consistency_score(full_text, lang),
                        "Backtracks": len(BACKTRACK_SIGNALS.findall(full_text)),
                        "Length": len(tokenizer.encode(full_text)) if tokenizer is not None else None,
                        "Format": format_reward(full_text),
                        "Question Correct Count": correct_count,
                    })
    return records