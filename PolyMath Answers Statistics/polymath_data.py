import argparse
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

# Defaults, used when the scripts are run with no CLI args at all.
MODEL_NAME = "XueZhang-bjtu/1.5B-cold-start-SFT"
RESULTS_DIR = "../evaluation/logs-eval/PolyMath-temp_0.9/1.5B-cold-start-SFT"
LEVELS = ["low", "medium", "high", "top"]
LANGS = ["en", "es", "fr", "pt"]
NUM_SAMPLES = 4

# All model result directories live under here, one subfolder per --model_name
# (e.g. "1.5B-cold-start-SFT", "MTHINKER", "MULE" — matches whatever --model_name
# was passed to polymath_res_gen.py during generation).
BASE_RESULTS_DIR_TEMPLATE = "../evaluation/logs-eval/PolyMath-temp_{temp}"


def parse_cli_args():
    """Common CLI args for Total_answers_stats.py / Per_question_stats.py, so you can point
    at any generated model's results without editing this file."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", default=None,
        help="Directory label under logs-eval/PolyMath-temp_<temp>/ to analyze, e.g. "
             "'1.5B-cold-start-SFT', 'MTHINKER', 'MULE'. Defaults to the SFT model.",
    )
    parser.add_argument(
        "--tokenizer", default=None,
        help="HF repo id or local path for the tokenizer used to count tokens. "
             "--model_name labels like 'MTHINKER'/'MULE' aren't valid HF ids themselves, so "
             "this defaults to the base SFT model's tokenizer, which is safe for LoRA/DPO "
             "checkpoints sharing the same vocab.",
    )
    parser.add_argument(
        "--results_dir", default=None,
        help="Explicit override for the results directory; if omitted, derived from "
             "--model_name and --temp.",
    )
    parser.add_argument("--temp", type=float, default=0.9)
    parser.add_argument("--levels", nargs="+", default=None)
    parser.add_argument("--langs", nargs="+", default=None)
    args = parser.parse_args()

    args.model_name = args.model_name or MODEL_NAME.split("/")[-1]
    args.tokenizer = args.tokenizer or MODEL_NAME
    args.results_dir = args.results_dir or os.path.join(
        BASE_RESULTS_DIR_TEMPLATE.format(temp=args.temp), args.model_name
    )
    args.levels = args.levels or LEVELS
    args.langs = args.langs or LANGS
    return args

# Extends the backtrack-signal regex from "RL Train Answers Statistics" with more phrases per
# language (incl. Spanish, previously missing) so detection density is comparable across all
# four scoped languages (en/es/fr/pt) rather than skewed toward whichever has more patterns.
BACKTRACK_SIGNALS = re.compile(
    # ============================= ENGLISH =============================
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
    r'\bre-?checking\b|\bthere(?:\'s| is) an error\b|'
    r"\bi(?:'m| am) not sure\b|\bi don't understand\b|"
    r'\b(?:look for|try) another (?:way|approach)\b|'
    r'\bno,\s*no\b|'
    r'\bhmm+\b|\bcorrection\b|\bmiscalculat(?:ed|ion)\b|'
    r"\bsomething(?:'s| is) wrong\b|\bdouble.?check\b|\brecheck\b|\bredo\b|"
    r'\bre-?(?:examine|calculate|compute|evaluate|verify|consider)\b|'
    r"\bthat can(?:no|')t be\b|"
    # ---- English additions (round 2) ----
    r'\balternatively\b|\bwrong\b|\bincorrect\b|\bnot sure\b|\bunsure\b|'
    r'\bnot (?:right|correct|quite)\b|\bcontradict(?:ion|s|ory)\b|'
    r'\binconsisten(?:t|cy)\b|\bconfusing\b|\bnope\b|\bnah\b|'
    r'\b(?:think again|rethink)\b|\bgo(?:ing)? back\b|\bback up\b|'
    r"\bthat doesn(?:'t| not) (?:seem|look|make)\b|"
    r"\bdoesn'?t make sense\b|\bmakes no sense\b|"
    r"\bthat(?:'s| is) (?:odd|strange|weird)\b|"
    r'\bi confused\b|\bi mixed up\b|\bmisread\b|\bmisunderst(?:ood|and)\b|'
    r'\bi think i (?:made|did|got)\b|\b(?:i|that) was wrong\b|\bthis is wrong\b|'
    r'\blet me fix\b|\bfix(?:ed|ing)? (?:this|that|it)\b|\bnever mind\b|'
    r'\blet me (?:verify|check|confirm|re-?examine|recalculate|recompute)\b|'
    r'\bre-?(?:examin|evaluat|calculat|comput|verif|consider|check|read)\w*\b|'
    r'\bre-?do\b|'
    # ============================= FRENCH ==============================
    r'\battends\b|\ben fait\b|\bnon,?\s+attendez\b|'
    r'\bje me suis tromp|\breprenons\b|'
    r'\boubl(?:ie|iez) ça\b|\blaisse-moi refaire\b|'
    r"(?:ce n'est pas correct|c'est faux)\b|\bmon erreur\b|"
    r"\bl'erreur (?:est|était) dans\b|"
    r"\bil y a une erreur\b|\bnon,\s*non\b|"
    r"\b(?:chercher|essayer) une autre (?:façon|méthode)\b|"
    r"\bje ne comprends pas\b|\bje ne suis pas s[ûu]r\b|"
    r'\bje dois (?:vérifier|revoir|recalculer|reconsidérer)\b|'
    r"\bc'est incorrect\b|\bce n'est pas (?:possible|juste|bon)\b|\bça ne peut pas\b|"
    r'\berreur dans\b|\ben réalité\b|\bje me trompe\b|\bje ne suis pas certain\b|'
    r"\bj'ai fait une erreur\b|"
    r'\b(?:vérifions|revérifions|recalculons|reconsidérons|recommençons|refaisons)\b|'
    r'\bun (?:moment|instant)\b|'
    # ---- French additions (round 2) ----
    r'\balternativement\b|\brefaire\b|\bje refais\b|\bje revois\b|\brecommencer\b|'
    r'\brevenons\b|\bje doute\b|\bje me suis planté\b|'
    r"\bj(?:e|')ai tort\b|\bça n(?:e|')a pas de sens\b|\bn(?:e|')a aucun sens\b|"
    r'\bça ne colle pas\b|\bça ne correspond pas\b|'
    r"\bc(?:e|')est (?:bizarre|étrange)\b|"
    # ============================= PORTUGUESE =========================
    r'\bespera\b|\bna verdade\b|\bnão,?\s+espera\b|'
    r'\bcometi um erro\b|\besquece isso\b|\bdeixa eu refazer\b|'
    r'\bisso está (?:errado|incorreto)\b|\bmeu erro\b|\bpensando bem\b|'
    r'\bo erro (?:está|estava) em\b|'
    r'\bhá um erro\b|\bnão,\s*não\b|'
    r'\b(?:procurar|tentar) outra forma\b|'
    r'\bnão entendo\b|\bnão tenho certeza\b|'
    r'\bde fato\b|\bnão pode ser\b|\bnão está (?:certo|correto)\b|\bestá errado\b|'
    r'\balgo está errado\b|\berro em\b|'
    r'\bvamos (?:verificar|revisar|recalcular|reconsiderar|conferir)\b|'
    r'\bpreciso (?:revisar|verificar|recalcular)\b|\bcorrig(?:ir|ido)\b|'
    r'\besper[ae]m\b|\bum (?:momento|segundo|instante)\b|'
    # ---- Portuguese additions (round 2) ----
    r'\balternativamente\b|\bme enganei\b|\benganei-me\b|\beu errei\b|\berrei\b|'
    r'\bfiz errado\b|\brefazer\b|\brefaço\b|\brevendo\b|\brevisando\b|\bopa\b|'
    r'\bnão faz sentido\b|\bnão (?:bate|fecha|confere|condiz|corresponde)\b|'
    r'\bnão estou convencido\b|'
    # ============================= SPANISH ============================
    r'\bespera\b|\ben realidad\b|\bno,?\s+espera\b|'
    r'\bcometí un error\b|\beso (?:está mal|es incorrecto)\b|'
    r'\bun momento\b|\bpensándolo bien\b|\bdéjame reconsiderar\b|'
    r'\bel error (?:está|estaba) en\b|'
    r'\bhay (?:un|una) error\b|'
    r'\b(?:buscar|intentar) otra manera\b|'
    r'\bno entiendo\b|\bno estoy segur[oa]\b|'
    r'\bde hecho\b|\bno es correcto\b|\bno puede ser\b|\balgo (?:está|anda) mal\b|'
    r'\b(?:esto|eso) no está bien\b|\berror en\b|\bme equivoqu[ée]\b|'
    r'\bdebo (?:revisar|verificar|recalcular)\b|'
    r'\b(?:verifiquemos|revisemos|comprobemos|recalculemos|reconsideremos)\b|'
    r'\bre-?(?:examinar|considerar)\b|\besper[ae]n\b|'
    # ---- Spanish additions (round 2) ----
    r'\balternativamente\b|\bincorrecto\b|\b(?:está |estar )?equivocad[oa]\b|'
    r'\bme he equivocado\b|\brevisando\b|\bhice mal\b|\bhay algo mal\b|'
    r'\bno (?:tiene sentido|coincide|cuadra|concuerda)\b',
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