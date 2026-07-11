from rewards import acc_compute_score, last_boxed_only_string, whether_cons, remove_boxed
import json
import pandas as pd
import random
import numpy as np
from collections import Counter
import re
import logging
import wandb
import argparse
from transformers import AutoTokenizer
import fasttext, re
from statistics import mean


DIFFICULTY_LOG_FILE = "./Datos/difficulty_log.jsonl"
MODEL_NAME = "XueZhang-bjtu/1.5B-cold-start-SFT"
ITERATION_NUMBER = 1

# D2 chosen-ranking weights: rep/uniq replace the old backtrack penalty (D1).
# bt(a) stays computed and logged as a diagnostic but never influences selection.
LAMBDA_REP = 0.3    # rep in [0,1], scaled x100 to be commensurate with chrF align [0,100]
MU_UNIQ = 1.0       # distinct 10-grams / 1000; a clean 8k-word answer earns ~8 points
REP_NGRAM = 10      # n-gram order for rep/uniq; keep identical everywhere

# D3 anti-looping pairs (thresholds calibrated on PolyMath high tier;
# M-Thinker in-language rep ~0.055, SFT/MuLE looping ~0.18-0.24)
POOL_A_REP_MAX = 0.08    # chosen side: clean progress
POOL_B_REP_MIN = 0.15    # rejected side: in-language answers that looped and failed
ANTILOOP_LEN_TOL = 0.25  # max relative length mismatch within a pair

# D4 census: "long" chosen candidates, in words (~4k tokens at ~1.3 tok/word)
CENSUS_MIN_WORDS = 3000
CENSUS_MIN_CELL = 50     # below this a (lang, difficulty) cell is flagged, not fabricated

# Hard ceiling on rep for chosen ELIGIBILITY (not just ranking): with the RPO
# anchor, a looping chosen is actively imitated, so a sole-correct-answer
# question must not promote a degenerate trace. Well above the correct-answer
# mean (~0.04-0.06) so it only removes genuine loops; costs only yield.
CHOSEN_REP_MAX = 0.15

# Trainer sequence cap (dpo_train_no_precomp.py --max_length). Pairs longer
# than this are front-truncated by truncation_mode="keep_end" (the PROMPT is
# amputated), so the filter counts them instead of discovering it in training.
TRAIN_MAX_LENGTH = 16384

# Prompt templates per language — same as generate_dpo_data.py
LANG_TO_INSTRUCTIONS = {
    'en': "{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}.",
    'es': "{question}\nPor favor, razona paso a paso y pon tu respuesta final dentro de \\boxed{{}}.",
    'fr': "{question}\nVeuillez raisonner étape par étape et mettre votre réponse finale dans \\boxed{{}}.",
    'pt': "{question}\nPor favor, raciocine passo a passo e coloque sua resposta final dentro de \\boxed{{}}.",
}

# ---------------------------------------------------------------------------
# Language consistency detection (windowed, restricted label space)
#
# fastText lid.176 scores overlapping windows. A window whose absolute top-1
# label is OUTSIDE the candidate set (e.g. Chinese leakage) keeps that label so
# it counts as off-language; otherwise confidence is renormalized over the four
# candidate languages {en, fr, pt, es} so short windows stay decidable.
# The label is three-way:
#    1 -> consistent:   chosen-eligible (no confirmed off-language content)
#    0 -> inconsistent: confident negative, usable as LC-contrast rejected
#   -1 -> ambiguous:    excluded from both pools (never trained on)
# ---------------------------------------------------------------------------
try:
    lid = fasttext.load_model("lid.176.bin")
except Exception:
    lid = None
    logging.warning("lid.176.bin not available. Language consistency check will be skipped.")

CANDIDATE_LANGS = ("en", "fr", "pt", "es")
WINDOW_SIZE = 350         # chars per detection window (LID is reliable at this size)
WINDOW_STRIDE = 175       # overlap so a genuine switch always spans >=2 windows
CONF_THRESHOLD = 0.8      # renormalized confidence below this -> ambiguous window
INCONSISTENT_FRAC = 0.30  # confirmed off-language window fraction -> confident negative
LEXICAL_EN_HITS = 3       # english function-word hits that disqualify a non-english chosen

# English function words that do not exist in fr/pt/es: catches single-word
# code-switches ("Wait, ...") that are below the window granularity.
EN_EXCLUSIVE = re.compile(
    r"\b(the|which|wait|should|would|could|because|therefore|answer|actually|"
    r"let's|we're|doesn't|isn't|there|where|then|thus|hence|since|both|"
    r"again|another|something|anything)\b",
    re.IGNORECASE,
)

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
    r'\bno (?:tiene sentido|coincide|cuadra|concuerda)\b|'
    # ======== round 3: data-mined from iter1_questions.jsonl ========
    # English (let's-see family: 29k/23k/12k hits missed by rounds 1-2)
    r"\blet'?s (?:see|check|verify|double.?check)\b|\blet me see\b|"
    r'\bi must have (?:made|mis)\w*\b|\bsanity check\b|\bseems off\b|\bam i missing\b|'
    # French (vérifie/je vérifie: 38k combined hits, the largest gap found)
    r'\bvérifie\b|\bmais non\b|\bvoyons\b|\bpardon\b|\bnon\s+non\b|'
    # Portuguese
    r'\bverificando\b|\bvou (?:verificar|checar|refazer)\b|\bvamos ver\b|'
    r'\bvejamos\b|\bcalma\b|\bops\b|\beita\b|\bespere\b|\bestranho\b|\bnão\s+não\b|'
    # Spanish (untested analogues of the mined fr/pt cues - no es data in iter1)
    r'\bveamos\b|\bvamos a ver\b|\bverifico\b|\bverificando\b',
    re.IGNORECASE
)

# copied from the MMATH eval — keep byte-identical so filter labels match the metric
def _strip_math(text):
        cleaned = re.sub(r'\$\$.*?\$\$|\\\(.*?\\\)|\\\[.*?\\\]', '', text, flags=re.DOTALL)
        cleaned = re.sub(r'\$[^$]*\$', '', cleaned)
        cleaned = re.sub(r'<[^>\n]{1,200}>', ' ', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned

_THINK_TAGS = re.compile(r"</?think>")

def rep_uniq(text, n=REP_NGRAM):
    """(rep, uniq, n_words) for one answer — spec §2-3.

    rep: fraction of word n-grams that duplicate an earlier n-gram in the same
    answer; language-neutral by construction (no word lists, no language ID).
    uniq: distinct n-grams / 1000 — novel content mass, offline chosen ranking
    ONLY (a policy can hack it online by padding with novel filler).
    Computed on the full text (thinking + final answer) with the <think> tags
    stripped; LaTeX is kept — repeated equation blocks are real loops.
    """
    words = _THINK_TAGS.sub(" ", text).split()
    total = len(words) - n
    if total <= 0:
        return 0.0, 0.0, len(words)
    grams = [tuple(words[i:i + n]) for i in range(total)]
    distinct = len(set(grams))
    return 1.0 - distinct / len(grams), distinct / 1000.0, len(words)

def _windows(text):
    if len(text) <= WINDOW_SIZE:
        return [text]
    return [text[i:i + WINDOW_SIZE]
            for i in range(0, len(text) - WINDOW_SIZE + WINDOW_STRIDE, WINDOW_STRIDE)]

def _classify_window(window):
    """Language code for the window, or None when the detector is unsure.

    The absolute top-1 label is checked BEFORE renormalizing over the candidate
    set: a window confidently in a non-candidate language (e.g. Chinese leakage,
    a known R1-distill failure mode) returns that code so it counts as an
    off-language window downstream, instead of having its residual mass
    reassigned to whichever candidate scraped into the top-20."""
    labels, probs = lid.predict(window, k=20)
    top_code = labels[0].replace("__label__", "")
    if top_code not in CANDIDATE_LANGS:
        # off-candidate language: trust the absolute top-1 when confident,
        # otherwise the window stays ambiguous
        return top_code if probs[0] >= CONF_THRESHOLD else None
    cand = {}
    for label, prob in zip(labels, probs):
        # languages in the window (between the candidates)
        code = label.replace("__label__", "")
        if code in CANDIDATE_LANGS:
            cand[code] = prob
    total = sum(cand.values())
    if total <= 0:
        return None
    best = max(cand, key=cand.get)
    if cand[best] / total < CONF_THRESHOLD:
        # no dominant language in this window (ambiguous) - don't count it as a confirmed off-language window
        return None
    return best

def _label_windows(text: str, expected_lang: str) -> int:
    '''
    Windowed three-way label for ONE section: 1 consistent / 0 inconsistent / -1 ambiguous.

    Off-language windows only count as confirmed when they form a run of >=2
    consecutive windows (with overlapping windows a genuine paragraph-level
    switch always spans at least two); isolated detections on single windows
    are treated as detector noise. A section short enough to produce a single
    window (e.g. the answer section) has no neighbour to confirm with, so a
    confident single-window detection counts.
    '''
    stripped = _strip_math(text)
    if len(stripped) < 40:
        return 1  # too short to judge - benefit of the doubt (mirrors the eval)

    labels = [_classify_window(w) for w in _windows(stripped)]
    n = len(labels)

    # confirm off-language windows by run length
    confirmed_off = 0
    i = 0
    while i < n:
        run_label = labels[i]
        j = i
        while j < n and labels[j] == run_label:
            j += 1
        if run_label is not None and run_label != expected_lang and (j - i >= 2 or n == 1):
            confirmed_off += j - i # count all windows in the run as confirmed off-language
        i = j

    n_target = sum(1 for lab in labels if lab == expected_lang)

    if confirmed_off / n >= INCONSISTENT_FRAC:
        # confidently inconsistent
        return 0
    if confirmed_off == 0 and n_target >= 0.5 * n:
        # confidently consistent
        return 1
    #  ambiguous
    return -1


def language_consistency_label(text: str, expected_lang: str) -> int:
    '''
    Per-section LC label mirroring the eval (cal-MMATH-acc.py): the text is
    split once at </think> and BOTH sections must be consistent, so a short
    off-language answer section vetoes the label even when it is a small
    fraction of the whole text. Combination of the section labels:
        either section 0  -> 0   (confident negative)
        else either -1    -> -1  (ambiguous, excluded from both pools)
        else              -> 1
    Single-word English insertions ("Wait, ...") are below window granularity,
    so they are caught by the EN_EXCLUSIVE lexical check over the whole text.
    '''
    if lid is None:
        return 1  # detector unavailable - benefit of the doubt (warned at load time)

    if "</think>" in text: # format only correct if there is just one closed think block
        think_pred, answer_pred = text.split("</think>", 1)
        section_labels = (_label_windows(think_pred, expected_lang),
                          _label_windows(answer_pred, expected_lang))
        if 0 in section_labels:
            combined = 0
        elif -1 in section_labels:
            combined = -1
        else:
            combined = 1
    else:
        # no closed think block - label the whole text
        combined = _label_windows(text, expected_lang)

    if combined == 1 and expected_lang != "en":
        # word-level code-switches, counted over the whole text
        if len(EN_EXCLUSIVE.findall(_strip_math(text))) >= LEXICAL_EN_HITS:
            return -1
    return combined
    
def format_reward(text):
    # Aligned with the eval (compute_score_acc_lc): exactly one </think> (its
    # split needs exactly 2 parts), len > 1000, and \boxed{} after the close.
    # No promptness condition: the eval never checks how soon the box appears,
    # and the model's median summary is ~1.4k chars (format_stats.py).
    if text.count('</think>') != 1 or len(text) <= 1000:
        return 0
    close_pos = text.rfind('</think>')
    if not any(m.start() > close_pos for m in re.finditer(r'\\boxed\{', text)):
        return 0
    return 1

def get_latex(text):
    """Extract LaTeX from the final reasoning segment only."""
    # split at backtrack signals, take last segment
    splits = [m.start() for m in BACKTRACK_SIGNALS.finditer(text)]
    final_segment = text[splits[-1]:] if splits else text

    results = []
    for p in [r'\\\((.+?)\\\)', r'\\\[(.+?)\\\]', r'\$([^\$\n]+)\$']:
        # for each of the LaTeX re's
        for item in re.findall(p, final_segment, re.DOTALL):
            # iterate through all latex chunks and add them to list
            results.append(item if isinstance(item, str) else ' '.join(item))
    
    # latex string (join everything) 
    s = ' '.join(results)
    # normalize - dfrac = frac
    s = re.sub(r'\\dfrac', r'\\frac', s)
    # remove displaystyle
    s = re.sub(r'\\displaystyle\s*', '', s)
    # remove \s
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def chrf(test_answer, ref_answer, n=6, beta=1):
    """Character n-gram F-score over LaTeX strings."""
    if not test_answer or not ref_answer:
        # one of the questions has no LaTeX
        return 0.0
    
    def get_ngrams(text, n):
        """Return n-grams from the text as a dictionary {n-gram:frequency}"""
        ng = {}
        for i in range(len(text) - n + 1):
            k = text[i:i+n]
            ng[k] = ng.get(k, 0) + 1
        return ng

    total_prec, total_rec = 0.0, 0.0
    for order in range(1, n + 1):
        # get 1-grams, 2-grams, ... and n-grams
        test_n = get_ngrams(test_answer, order)
        ref_n = get_ngrams(ref_answer, order)
        if not test_n or not ref_n:
            # no order-grams
            continue
        # number of appearances of common n-grams
        m = sum(min(test_n.get(g, 0), ref_n.get(g, 0)) for g in ref_n)
        # precision per order
        total_prec += m / sum(test_n.values())
        # recall per order
        total_rec += m / sum(ref_n.values())
    # average precision and recall per order
    avg_prec, avg_rec = total_prec / n, total_rec / n
    if avg_prec + avg_rec == 0:
        # no common n-grams
        return 0.0
    # compute chrF score (F-score of n-gram overlap)
    return 100 * (1 + beta**2) * avg_prec * avg_rec / (beta**2 * avg_prec + avg_rec)

def get_k(correct_rate_lang, total_consensus, n_answers):
    if correct_rate_lang >= 0.75 * n_answers and total_consensus >= 0.75:
        return 1
    elif correct_rate_lang >= 0.5 * n_answers or total_consensus >= 0.5:
        return 2
    else:
        return 3
    
def select_rejected(lang, rejected_wrong_ans, rejected_inconsistent_ans, other_rejected, k):
    if lang == "en":
        priority_pool = rejected_wrong_ans
        secondary_pool = rejected_inconsistent_ans
    else:
        priority_pool = rejected_inconsistent_ans
        secondary_pool = rejected_wrong_ans
    
    selected = []
    
    # sample from priority pool first, without replacement
    n_from_priority = min(k, len(priority_pool))
    selected += random.sample(priority_pool, n_from_priority)
    
    # fill remaining from secondary pool
    remaining = k - len(selected)
    if remaining > 0:
        n_from_secondary = min(remaining, len(secondary_pool))
        selected += random.sample(secondary_pool, n_from_secondary)
    
    # fill remaining from other rejected
    remaining = k - len(selected)
    if remaining > 0:
        selected += random.sample(other_rejected, min(remaining, len(other_rejected)))
    
    return selected

def select_rejected2(lang, rejected_wrong_ans, rejected_inconsistent_ans, other_rejected, k):
    priority_pool = rejected_wrong_ans + rejected_inconsistent_ans
    secondary_pool = other_rejected
    
    selected = []
    
    # sample from priority pool
    n_from_priority = min(k, len(priority_pool))
    selected += random.sample(priority_pool, n_from_priority)
    
    # fill remaining from secondary pool
    remaining = k - len(selected)
    if remaining > 0:
        n_from_secondary = min(remaining, len(secondary_pool))
        selected += random.sample(secondary_pool, n_from_secondary)
    
    return selected


def select_rejected_mixed(lang, rejected_wrong_ans, rejected_inconsistent_ans,
                          other_rejected, k, chosen_ids, answers):
    """Quota-based rejected selection (replaces the priority-fill select_rejected).

    Non-English languages need BOTH signals, and the old priority starved the
    accuracy signal exactly where it was most available (acc-contrast pools are
    ~3x more common than lc-contrast ones in iter1):
      k == 1 -> one accuracy-contrast pair (in-language wrong answer) when
                available, falling back to lc-contrast (easy questions rarely
                need language pressure - LC at low tier was already ~0.98);
      k >= 2 -> guarantee one lc-contrast AND one accuracy-contrast pair when
                both pools are non-empty, fill the remainder accuracy-first.
    English keeps accuracy priority (its inconsistent pool is empty in practice).

    lc-contrast picks are length-matched to the chosen answer they will be
    zipped with: correct-but-English answers are ~4x longer than chosen ones
    on average, and random pairing would stack a length signal on top of the
    language signal.

    Returns a list positionally aligned with chosen_ids (the caller zips them).
    """
    k = min(k, len(chosen_ids)) if chosen_ids else 0
    if k == 0:
        return []

    if lang == "en" or k == 1:
        # accuracy-contrast first, lc-contrast as fallback
        n_acc = min(k, len(rejected_wrong_ans))
        n_lc = min(k - n_acc, len(rejected_inconsistent_ans))
    else:
        # guarantee one of each when both pools are non-empty
        n_lc = min(1, len(rejected_inconsistent_ans))
        n_acc = min(k - n_lc, len(rejected_wrong_ans))
        # remainder: lc pool before the junk pool
        n_lc += min(k - n_acc - n_lc, len(rejected_inconsistent_ans) - n_lc)
    n_other = min(k - n_acc - n_lc, len(other_rejected))

    selected = random.sample(rejected_wrong_ans, n_acc)

    # length-matched lc picks, aligned with the chosen ids they will pair with
    lc_pool = list(rejected_inconsistent_ans)
    for c_id in chosen_ids[n_acc:n_acc + n_lc]:
        target_len = len(answers[c_id])
        best = min(lc_pool, key=lambda r: abs(len(answers[r]) - target_len))
        lc_pool.remove(best)
        selected.append(best)

    selected += random.sample(other_rejected, n_other)
    return selected


def _dedup_leading_think(prompt: str, completion: str) -> str:
    """Stored candidates carry a manually prepended "<think>\\n" (generate_dpo_data.py),
    and the chat template's generation prompt may also end with the opening tag.
    Strip the completion's leading tag ONLY when the prompt already provides one,
    so the training sequence never contains "<think>\\n<think>\\n" — and a candidate
    keeps its own tag if the template does not add it."""
    if not prompt.rstrip().endswith("<think>"):
        return completion
    if completion.startswith("<think>\n"):
        return completion[len("<think>\n"):]
    if completion.startswith("<think>"):
        return completion[len("<think>"):]
    return completion


def dataset_priorities(candidates, tokenizer, n_answers, dataset_file):
    # n_answers is per language; totals across the three languages use n_answers_total
    n_answers_total = 3 * n_answers

    # define dictionaries to store useful information
    en_scores = dict()
    fr_scores = dict()
    pt_scores = dict()
    scores = dict({"en": en_scores, "fr": fr_scores, "pt": pt_scores})
    langs = ["en", "fr", "pt"]
    pivot_langs = dict()
    correct_rate = dict()
    acc_rate = dict()
    total_consensus = dict()
    #pair_score = dict()
    chosen_candidates = dict({"en": dict(), "fr": dict(), "pt": dict()})
    rejected_candidates = dict({"en": dict(), "fr": dict(), "pt": dict()})
    chrF_alignment = dict()
    ids_with_pairs = set()
    all_answers = dict()
    rep_by_lang = {"en": [], "fr": [], "pt": []}       # per-answer rep, for stats
    chosen_rep_by_lang = {"en": [], "fr": [], "pt": []}  # rep of chosen-eligible answers

    # list to store the pairs for DPO
    dpo_pairs = []
    
    # for logging
    stats = {
        "total_questions": len(candidates)//3, #target_candidates is the total number of questions (counting the three languages as 1 question)
        "skipped_wrong_dataset_id": 0, # error in dataset: mismatched id's
        "skipped_wrong_dataset_lang": 0, # error in dataset: wrong languages
        "skipped_no_ground_truth": 0, # error in dataset: missing ground truth
        "skipped_no_correct_answers": 0, # model did not manage to find correct answer in any language for a specific question
        "skipped_no_chosen": 0, # model did not manage to find correct answers in that language for a sepcific question
        "skipped_no_rejected": 0, # all model answers correct
        "pairs_created": 0, # pair created    
        "question_lang_with_pairs": 0, # question-lang combination with at least one pair
        "chosen_correct": 0, # chosen questions
        "rejected_incorrect": 0, # rejected questions
        "excluded_correct_ambiguous_lang": 0, # correct answers with ambiguous language - never trained on
        "excluded_correct_bad_format": 0, # correct in-language answers with broken format - never trained on
        "excluded_correct_high_rep": 0, # correct in-language answers that loop (rep >= CHOSEN_REP_MAX) - never trained on
        "lc_consistent": 0, # answers confidently in the expected language
        "lc_inconsistent": 0, # answers confidently in another language
        "lc_ambiguous": 0, # answers the detector could not label confidently
        "skipped_no_correct_otherlang_answers": 0, # skipped pivot language question because none of the other languages had a correct answer 
        "skipped_no_correct_answers_ids": [], # id's of questions with no correct answers in any language
        "skipped_no_correct_answers_ids_langs": [], # id-lang pairs of questions with no correct answers in that language
        "english_pivot": 0, # number of times English was selected as pivot
        "french_pivot": 0, # number of times French was selected as pivot
        "portuguese_pivot": 0 # number of times Portuguese was selected as pivot
    }
    # realized pair-type mixture per language, keyed by the rejected side:
    # acc_contrast (wrong, in-language), lc_contrast (correct, off-language), other
    for _lang in langs:
        for _ptype in ("acc_contrast", "lc_contrast", "other", "antiloop"):
            stats[f"pairs_{_lang}_{_ptype}"] = 0
    # D3: (q, lang) combos where both anti-loop pools existed but no pairing
    # satisfied the +-25% length-match constraint
    stats["antiloop_skipped_length_mismatch"] = 0
    # pairs whose prompt+completion exceeds the trainer cap (keep_end would
    # amputate the prompt at training time)
    stats["pairs_over_cap_chosen"] = 0
    stats["pairs_over_cap_rejected"] = 0
    stats["pairs_over_cap_any"] = 0

    print("Calculating answer scores...")
    for c_i in range(0, len(candidates), 3):
        print(f"Processing question {c_i//3 + 1}/{len(candidates)//3} (index {c_i})")
        # get answers for three languages
        if len(candidates) <= c_i + 2:
            continue
        en_answers = candidates[c_i]
        fr_answers = candidates[c_i + 1]
        pt_answers = candidates[c_i + 2]
        
        # id's differ - error (skip)
        if en_answers["numerical_id"] != fr_answers["numerical_id"] or en_answers["numerical_id"] != pt_answers["numerical_id"]:
            print(f"ID mismatch at index {c_i}: {en_answers['numerical_id']}, {fr_answers['numerical_id']}, {pt_answers['numerical_id']}")
            stats["skipped_wrong_dataset_id"] += 1
            continue
            
        # languages invalid - error (skip)
        if en_answers["language"] != "en" or fr_answers["language"] != "fr" or pt_answers["language"] != "pt":
            print(f"Language mismatch at index {c_i}: {en_answers['language']}, {fr_answers['language']}, {pt_answers['language']}")
            stats["skipped_wrong_dataset_lang"] += 1
            continue
        
        # no ground truth - error (skip)
        if en_answers["ground_truth"] is None or fr_answers["ground_truth"] is None or pt_answers["ground_truth"] is None or en_answers["ground_truth"] != fr_answers["ground_truth"] or en_answers["ground_truth"] != pt_answers["ground_truth"]:
            print(f"Missing ground truth at index {c_i}: {en_answers['ground_truth']}, {fr_answers['ground_truth']}, {pt_answers['ground_truth']}")
            stats["skipped_no_ground_truth"] += 1
            continue
        
        # get numerical id of the question
        num_id = en_answers["numerical_id"]
        # update answers dict
        all_answers[num_id] = [en_answers, fr_answers, pt_answers]
        
        # initialize dicts for this question
        correct_rate[num_id] = dict()
        acc_rate[num_id] = dict()
        
        for i in range(3):
            # dictionary to store scores for that language
            scores_lang = scores[langs[i]]
            scores_lang[num_id] = dict()
            # get the answers for that language
            lang_answers = all_answers[num_id][i]
            # get the language code
            lang = langs[i]
            
            # initialize dicts for that language
            correct_rate[num_id][lang] = 0
            acc_rate[num_id][lang] = 0
            
            for j in range(n_answers):
                # get the answer
                answer = lang_answers["candidates"][j]
                
                # scores for that answer
                scores_lang[num_id][j] = dict()
                
                # get accuracy
                acc = acc_compute_score(answer, lang_answers["ground_truth"])
                scores_lang[num_id][j]["acc"] = acc
                
                # get language consistency (1 consistent / 0 inconsistent / -1 ambiguous)
                lc = language_consistency_label(answer, lang)
                scores_lang[num_id][j]["lc"] = lc
                if lc == 1:
                    stats["lc_consistent"] += 1
                elif lc == 0:
                    stats["lc_inconsistent"] += 1
                else:
                    stats["lc_ambiguous"] += 1
                
                # get format rewards
                fmt = format_reward(answer)
                scores_lang[num_id][j]["format"] = fmt
                
                # get backtrack count (diagnostic only — never scores, never
                # compared across languages: it measures code-switched English)
                scores_lang[num_id][j]["bt"] = len(BACKTRACK_SIGNALS.findall(answer))

                # repetition metrics (spec §2-3): duplicate 10-gram rate,
                # novel content mass, and word count (D4 census)
                rep_val, uniq_val, n_words = rep_uniq(answer)
                scores_lang[num_id][j]["rep"] = rep_val
                scores_lang[num_id][j]["uniq"] = uniq_val
                scores_lang[num_id][j]["n_words"] = n_words
                rep_by_lang[lang].append(rep_val)

                if acc == 1:
                    acc_rate[num_id][lang] += 1
                    if lc == 1 and fmt == 1 and rep_val < CHOSEN_REP_MAX:
                        # correct_rate counts chosen-eligible answers only, so
                        # pivot selection / consensus / get_k stay consistent
                        # with the chosen pool (acc, lc, format AND rep cap)
                        correct_rate[num_id][lang] += 1
    
    # iterate through questions to calculate pivot and pairwise scores     
    print("Selecting correct and incorrect answers")  
    for id in en_scores:      
        print(f"Processing question {id}")
        # total consensus score for the question (combining three languages)
        total_consensus[id] = sum([correct_rate[id][lang] for lang in langs])/n_answers_total
        
        # init chrf alignment dict
        chrF_alignment[id] = dict()
        
        if total_consensus[id] == 0:
            # no correct answers - not useful for training
            pivot_langs[id] = None
            stats["skipped_no_correct_answers"] += 1
            stats["skipped_no_correct_answers_ids"].append(id)
            continue
        
        # select pivot language as the one with highest correctness
        # in case of draw, this selects first language (which is English) - makes sense
        pivot_langs[id] = langs[np.argmax([correct_rate[id][lang] for lang in langs])]
        
        # update pivot stats
        if pivot_langs[id] == "en":
            stats["english_pivot"] += 1
        if pivot_langs[id] == "fr":
            stats["french_pivot"] += 1
        if pivot_langs[id] == "pt":
            stats["portuguese_pivot"] += 1
        
        # iterate through languages
        for lang in langs:
            chosen_candidates[lang][id] = []
            rejected_candidates[lang][id] = []
            # iterate though answers in that language
            for i in range(n_answers):
                # scores for that answer
                answer_scores = scores[lang][id][i]
                if (answer_scores["acc"] == 1 and answer_scores["lc"] == 1
                        and answer_scores["format"] == 1
                        and answer_scores["rep"] < CHOSEN_REP_MAX):
                    # correct, in-language, eval-scoreable format, not looping
                    # (format costs ~2% of correct answers - see format_stats.py)
                    chosen_candidates[lang][id].append(i)
                    stats["chosen_correct"] += 1
                    chosen_rep_by_lang[lang].append(answer_scores["rep"])
                elif (answer_scores["acc"] == 1 and answer_scores["lc"] == 1
                        and answer_scores["format"] == 1):
                    # correct, in-language, scoreable — but a degenerate loop:
                    # the RPO anchor would IMITATE it (a rank penalty cannot
                    # save a sole-correct-answer question). Don't imitate it,
                    # don't push it down (it is still correct reasoning).
                    stats["excluded_correct_high_rep"] += 1
                elif answer_scores["acc"] == 1 and answer_scores["lc"] == 1:
                    # correct and in-language but broken format: don't imitate it,
                    # don't push it down
                    stats["excluded_correct_bad_format"] += 1
                elif answer_scores["acc"] == 1 and answer_scores["lc"] == -1:
                    # correct but language ambiguous: don't imitate it, don't push it down
                    stats["excluded_correct_ambiguous_lang"] += 1
                else:
                    # incorrect answer
                    rejected_candidates[lang][id].append(i)
                    stats["rejected_incorrect"] += 1
            
        
        for lang in langs:
            if not chosen_candidates[lang][id]:
                # no correct answers in that language
                stats["skipped_no_chosen"] += 1
                stats["skipped_no_correct_answers_ids_langs"].append((id, lang))
                continue
            
            if not rejected_candidates[lang][id]:
                # no incorrect answers in that language
                stats["skipped_no_rejected"] += 1
                continue

            if lang != pivot_langs[id]:
                # get correct indexes of pivot language
                correct_pivot = chosen_candidates[pivot_langs[id]][id]
                
                # get answers in pivot language
                pivot_answers = [all_answers[id][langs.index(pivot_langs[id])]["candidates"][j] for j in correct_pivot]

                # not pivot lang - compare to pivot
                chrF_alignment[id][lang] = dict()
                # iterate through correct answers
                for c_id in chosen_candidates[lang][id]:
                    # get answer
                    answer = all_answers[id][langs.index(lang)]["candidates"][c_id]

                    # get latex of answer
                    latex_answer = get_latex(answer)
                    
                    # chrF scores comparing latex of candidate to latex of pivot answers
                    chrF_scores = [chrf(latex_answer, get_latex(ans)) for ans in pivot_answers]
                    # chrF alignment as maximum closeness between the answer and a (correct) pivot answer
                    chrF_alignment[id][lang][c_id] = max(chrF_scores) if chrF_scores else 0
            else: 
                # get correct indexes of other languages
                languages = langs.copy()
                languages.remove(lang)
                correct_answers = [all_answers[id][langs.index(l)]["candidates"][j] for l in languages for j in chosen_candidates[l][id]]
                
                if not correct_answers:
                    # no correct answers in other languages - skip question in pivot language
                    stats["skipped_no_correct_otherlang_answers"] += 1
                    continue
                 
                # for pivot language - compare it with the rest
                chrF_alignment[id][lang] = dict()
                for c_id in chosen_candidates[lang][id]:
                    # get answer
                    answer = all_answers[id][langs.index(lang)]["candidates"][c_id]
                    # get latex 
                    latex_answer = get_latex(answer)
                    
                    # chrF scores comparing latex of candidate to latex of pivot answers
                    chrF_scores = [chrf(latex_answer, get_latex(ans)) for ans in correct_answers]
                    # chrF alignment as maximum closeness between the answer and a (correct) pivot answer
                    chrF_alignment[id][lang][c_id] = max(chrF_scores) if chrF_scores else 0
                    
    # difficulty log
    print("Writing difficulty log...")

    difficulty_log = []

    def _difficulty_label(consensus):
            if consensus == 0:
                return "too_hard"   # model never got it right in any language
            elif consensus < 0.25:
                return "hard"       # rare correct answers
            elif consensus < 0.6:
                return "medium"     # sometimes correct — best DPO territory
            else:
                return "easy"       # mostly correct — weak DPO signal

    # D4 census: per (lang, difficulty), how many answers are eligible anti-loop
    # chosen candidates (acc & lc & format, clean rep, long). Cells below
    # CENSUS_MIN_CELL are recorded as a shortfall, never fabricated — the
    # shortfall itself bounds what offline DPO can achieve there.
    census = {lang: {"too_hard": 0, "hard": 0, "medium": 0, "easy": 0}
              for lang in langs}

    for id in en_scores:
        # skip questions that were skipped in scoring
        if id not in total_consensus:
            continue

        tier = _difficulty_label(total_consensus[id])
        for lang in langs:
            for j in range(n_answers):
                s = scores[lang][id][j]
                if (s["acc"] == 1 and s["lc"] == 1 and s["format"] == 1
                        and s["rep"] < POOL_A_REP_MAX
                        and s["n_words"] > CENSUS_MIN_WORDS):
                    census[lang][tier] += 1

        entry = {
            "num_id": id,
            "iteration": ITERATION_NUMBER,

            # overall difficulty
            "total_consensus": round(total_consensus[id], 4),
            "difficulty": _difficulty_label(total_consensus[id]),

            # per-language correct rate (acc + lc + format) out of n_answers
            "correct_rate_en": round(correct_rate[id]["en"] / n_answers, 4),
            "correct_rate_fr": round(correct_rate[id]["fr"] / n_answers, 4),
            "correct_rate_pt": round(correct_rate[id]["pt"] / n_answers, 4),

            # per-language accuracy rate (acc only, ignoring lc) out of n_answers
            "acc_rate_en": round(acc_rate[id]["en"] / n_answers, 4),
            "acc_rate_fr": round(acc_rate[id]["fr"] / n_answers, 4),
            "acc_rate_pt": round(acc_rate[id]["pt"] / n_answers, 4),

            # per-language mean 10-gram repetition rate over the n answers
            "rep_mean_en": round(mean(scores["en"][id][j]["rep"] for j in range(n_answers)), 4),
            "rep_mean_fr": round(mean(scores["fr"][id][j]["rep"] for j in range(n_answers)), 4),
            "rep_mean_pt": round(mean(scores["pt"][id][j]["rep"] for j in range(n_answers)), 4),

            # which language was strongest for this question
            "pivot_lang": pivot_langs[id],

            # whether pairs were actually generated for this question
            # filled in after third loop — placeholder for now
            "pairs_generated": False,
        }
        difficulty_log.append(entry)

    # iterate to generate the final pairs
    with open(dataset_file, "w", encoding="utf-8") as f:
        print("Generating pairs...")
        for candidate in candidates:
            id = candidate["numerical_id"]
            lang = candidate["language"]
            
            # scores for that question
            if id not in scores[lang]:
                # no answer for that question in that language - dataset error
                continue

            if id not in total_consensus or total_consensus[id] == 0:
                # no correct answers for that question
                continue

            if not chosen_candidates[lang][id] or not rejected_candidates[lang][id]:
                # no correct or incorrect answers - useless for training
                continue
            
            question_scores = scores[lang][id]
            
            # rejected candidate pools (shared by both pair types)
            rejected_cands = rejected_candidates[lang][id]
            # answers with wrong result but correct language consistency
            rejected_wrong_ans = [i for i in rejected_cands if question_scores[i]["acc"]==0 and question_scores[i]["lc"]==1]
            # answers with correct result but incorrect language consistency
            rejected_inconsistent_ans = [i for i in rejected_cands if question_scores[i]["acc"]==1 and question_scores[i]["lc"]==0]
            # other rejected ones (wrong answer with inconsistent or ambiguous language)
            other_rejected =  [i for i in rejected_cands if question_scores[i]["acc"]==0 and question_scores[i]["lc"]!=1]

            emitted = set()  # (chosen, rejected) index pairs written for this (q, lang)

            def emit_pair(chosen, rejected, ptype):
                # prompt
                prompt_text = LANG_TO_INSTRUCTIONS[lang].format(question=candidate["question"])
                messages = [{"role": "user", "content": prompt_text}]
                prompt_formatted = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=True,
                )

                # drop the completion's leading <think> only when the templated
                # prompt already ends with one (see _dedup_leading_think)
                pair_dict = {
                    "prompt": prompt_formatted,
                    "chosen": _dedup_leading_think(prompt_formatted, candidate["candidates"][chosen]),
                    "rejected": _dedup_leading_think(prompt_formatted, candidate["candidates"][rejected])
                }
                # over-cap monitor: with truncation_mode="keep_end", a pair
                # longer than the trainer cap trains on an amputated prompt.
                # Counted here so the problem is visible before training.
                over_any = False
                for side in ("chosen", "rejected"):
                    n_tok = len(tokenizer(prompt_formatted + pair_dict[side]).input_ids)
                    if n_tok > TRAIN_MAX_LENGTH:
                        stats[f"pairs_over_cap_{side}"] += 1
                        over_any = True
                if over_any:
                    stats["pairs_over_cap_any"] += 1

                # log the realized pair type (mixture monitoring)
                stats[f"pairs_{lang}_{ptype}"] += 1
                dpo_pairs.append(pair_dict)
                ids_with_pairs.add(id)
                emitted.add((chosen, rejected))
                stats["pairs_created"] += 1
                f.write(json.dumps(pair_dict, ensure_ascii=False) + "\n")

            # ---- pair type 1: quota-mixed acc/lc-contrast pairs (needs chrF pivot) ----
            if lang in chrF_alignment[id]:
                # dynamically select number of pairs to generate per question and language
                k = get_k(correct_rate[id][lang], total_consensus[id], n_answers)

                # D2 chosen ranking: alignment, minus verbatim recycling, plus
                # novel content mass — prefers "reasoned long and productively"
                # over both "short lucky hit" and "long loop". No backtrack term
                # (D1: bt ~ 0 on every lc==1 candidate, so it was blind exactly
                # where it applied; kept as a logged diagnostic only). No format
                # term: format is a hard gate for chosen, constant 1 here.
                def candidate_score(c_id):
                    s = question_scores[c_id]
                    return (chrF_alignment[id][lang][c_id]
                            - LAMBDA_REP * 100 * s["rep"]
                            + MU_UNIQ * s["uniq"])

                sorted_chosen = sorted(
                    chrF_alignment[id][lang].keys(),
                    key=candidate_score,
                    reverse=True
                )

                # take top k, but don't exceed available candidates
                top_k_chosen = sorted_chosen[:min(k, len(sorted_chosen))]

                # rejected answers (quota-mixed, length-matched lc picks)
                rejected_selected = select_rejected_mixed(
                    lang,
                    rejected_wrong_ans,
                    rejected_inconsistent_ans,
                    other_rejected,
                    k,
                    top_k_chosen,
                    candidate["candidates"]
                )

                for chosen, rejected in zip(top_k_chosen, rejected_selected):
                    r_scores = question_scores[rejected]
                    if r_scores["acc"] == 0 and r_scores["lc"] == 1:
                        ptype = "acc_contrast"
                    elif r_scores["acc"] == 1 and r_scores["lc"] == 0:
                        ptype = "lc_contrast"
                    else:
                        ptype = "other"
                    emit_pair(chosen, rejected, ptype)

            # ---- pair type 2 (D3): length-matched anti-looping pair, at most 1 ----
            # Clean "progress vs looping" gradient: both sides in-language and
            # length-matched within +-25%, so the dominant remaining difference
            # is looping vs progress.
            # Needs no chrF pivot, so it also fires on pivot-language questions
            # that type 1 skips (no correct other-language answer).
            pool_a = [i for i in chosen_candidates[lang][id]
                      if question_scores[i]["rep"] < POOL_A_REP_MAX]
            pool_b = [i for i in rejected_wrong_ans
                      if question_scores[i]["rep"] > POOL_B_REP_MIN]
            cand_pairs = [(a, b) for a in pool_a for b in pool_b if (a, b) not in emitted]
            if cand_pairs:
                def _len_mismatch(pair):
                    la = len(candidate["candidates"][pair[0]])
                    lb = len(candidate["candidates"][pair[1]])
                    return abs(la - lb) / max(la, lb)
                best = min(cand_pairs, key=_len_mismatch)
                if _len_mismatch(best) <= ANTILOOP_LEN_TOL:
                    emit_pair(best[0], best[1], "antiloop")
                else:
                    stats["antiloop_skipped_length_mismatch"] += 1

            if emitted:
                stats["question_lang_with_pairs"] += 1
    
    print("Finishing difficulty log...")
    # prompts with pairs
    for entry in difficulty_log:
        # change to true if entry["num_id"] in ids_with_pairs
        if entry["num_id"] in ids_with_pairs:
            entry["pairs_generated"] = True

    # rep summary + D4 census into stats (tracked in W&B via stats.json)
    for lang in langs:
        stats[f"rep_mean_{lang}"] = round(mean(rep_by_lang[lang]), 4) if rep_by_lang[lang] else None
        stats[f"chosen_rep_mean_{lang}"] = round(mean(chosen_rep_by_lang[lang]), 4) if chosen_rep_by_lang[lang] else None
    stats["pairs_over_cap_frac"] = (
        round(stats["pairs_over_cap_any"] / stats["pairs_created"], 4)
        if stats["pairs_created"] else None
    )
    print(f"pairs over the {TRAIN_MAX_LENGTH}-token training cap: "
          f"{stats['pairs_over_cap_any']}/{stats['pairs_created']} "
          f"(frac {stats['pairs_over_cap_frac']}; "
          f"chosen side {stats['pairs_over_cap_chosen']}, "
          f"rejected side {stats['pairs_over_cap_rejected']})")

    stats["antiloop_census"] = census
    print("D4 anti-loop census (eligible long clean chosen candidates):")
    for lang in langs:
        for tier, count in census[lang].items():
            flag = "  <-- SHORTFALL (recorded, not fabricated)" if count < CENSUS_MIN_CELL else ""
            print(f"  {lang}/{tier}: {count}{flag}")

    return dpo_pairs, stats, difficulty_log
 
    
def main():
    #try:
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates_file", type=str, default="./Datos/iter1_questions.jsonl")
    parser.add_argument("--n_answers", type=int, default=8)
    parser.add_argument("--dataset_file", type=str, default="./Datos/dpo_dataset.jsonl")
    parser.add_argument("--stats_file", type=str, default="./Datos/stats.json")
    parser.add_argument("--difficulty_log_file", type=str, default=DIFFICULTY_LOG_FILE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_questions", type=int, default=None,
                        help="Test runs: only process the first N questions (N triples of en/fr/pt lines).")

    args = parser.parse_args()

    # seed the rejected-candidate sampling so the generated dataset is reproducible
    random.seed(args.seed)


    counter = 0
    candidates = []
    # load answers file
    with open(args.candidates_file, "r", encoding="utf-8") as f:
        for line in f:
            counter += 1
            if line:
                try:
                    # Attempt to parse the individual JSON object
                    candidates.append(json.loads(line))
                except json.JSONDecodeError as e:
                    # Log the error but keep going!
                    print(f"Skipping line {counter} due to JSON error: {e}")
                    continue
    print("loaded file")

    if args.max_questions is not None:
        # test runs: keep only the first N questions (each question = 3 lines, en/fr/pt)
        candidates = candidates[:3 * args.max_questions]
        print(f"TEST MODE: limited to the first {args.max_questions} questions ({len(candidates)} lines)")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    dpo_pairs, stats, difficulty_log = dataset_priorities(candidates, tokenizer, args.n_answers, args.dataset_file)
       
    with open(args.stats_file, "w") as f:
        json.dump(stats, f)
        
    with open(args.difficulty_log_file, "w", encoding="utf-8") as f:
        for entry in difficulty_log:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        

    #except Exception as e:
    #    print(counter)
    #    print(f"Error: {e}")
    #    return      

if __name__ == "__main__":
    main()