from rewards import acc_compute_score, last_boxed_only_string, whether_cons, remove_boxed
import json
import pandas as pd
import random
import numpy as np
from collections import Counter
import re
import logging
import wandb
from transformers import AutoTokenizer

CANDIDATES_FILE = "./Datos/iter1_questions.jsonl"
N_ANSWERS = 8
DATASET_FILE = "./Datos/dpo_dataset.jsonl"
STATS_FILE = "./Datos/stats.json"
DIFFICULTY_LOG_FILE = "./Datos/difficulty_log.jsonl"
MODEL_NAME = "XueZhang-bjtu/1.5B-cold-start-SFT"
ITERATION_NUMBER = 1

WEIGHT_BACKTRACK = 2
WEIGHT_FORMAT = 5
SUMMARY_LENGTH = 800

# Prompt templates per language — same as generate_dpo_data.py
LANG_TO_INSTRUCTIONS = {
    'en': "{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}.",
    'es': "{question}\nPor favor, razona paso a paso y pon tu respuesta final dentro de \\boxed{{}}.",
    'fr': "{question}\nVeuillez raisonner étape par étape et mettre votre réponse finale dans \\boxed{{}}.",
    'pt': "{question}\nPor favor, raciocine passo a passo e coloque sua resposta final dentro de \\boxed{{}}.",
}

# load langdetect
try:
    from langdetect import detect, DetectorFactory, LangDetectException
    DetectorFactory.seed = 42
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False
    logging.warning("langdetect not installed. Language consistency check will be skipped.")
 

def is_language_consistent(text:str, expected_lang:str) -> bool:
    '''
    Check whether the text is written in the expected language, mirroring the LC reward
    LC(x) = (only one language detected) AND (that language == expected_lang)
    
    We only look at the <think>...</think> section and the answer section.
    Mathematical LaTeX doesn't count as a separate language, so we strip it first.
    '''
    if not LANGDETECT_AVAILABLE:
        return True  # skip check if langdetect not installed
    
    # Strip LaTeX-heavy sections to avoid confusing the language detector.
    # The model outputs: <think>...reasoning...</think> answer text \boxed{...}
    # We check the reasoning part.

    # Remove everything inside \boxed{...}
    text_for_detection = re.sub(r'\\boxed\{[^}]*\}', '', text)
    # Remove other LaTeX commands
    text_for_detection = re.sub(r'\$[^$]*\$', '', text_for_detection)
    text_for_detection = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', '', text_for_detection)
    # Remove think tags
    text_for_detection = re.sub(r'<think>|</think>', '', text_for_detection)
    text_for_detection = text_for_detection.strip()

    if len(text_for_detection) < 20:
    # Too short to reliably detect language — give benefit of the doubt
        return True
    
    try:
        detected = detect(text_for_detection)
        return detected == expected_lang
    except LangDetectException:
        return False
    
def format_reward(text):
    # Condition 1: think tags present
    if not re.search(r'<think>', text):
        return 0
    if not re.search(r'</think>', text):
        return 0

    # Condition 2: \boxed{} exists outside </think>
    close_pos = text.rfind('</think>')
    boxes_after = [m.start() for m in re.finditer(r'\\boxed\{', text)
                   if m.start() > close_pos]
    if not boxes_after:
        return 0

    # Condition 3: \boxed{} appears promptly after </think>, not buried
    if (min(boxes_after) - close_pos) > SUMMARY_LENGTH:
        return 0

    return 1

BACKTRACK_SIGNALS = re.compile(
    r'\bwait\b|\bactually\b|\bno,?\s+wait\b|'
    r'\blet me reconsider\b|\blet me restart\b|'
    r"that'?s wrong\b|\bi made an error\b|"
    r'\blet me try again\b|\bhmm,?\s+actually\b|'
    r'\bhold on\b|\bon second thought\b|'
    # French
    r'\battends\b|\ben fait\b|\bnon,?\s+attendez\b|'
    r'\bje me suis tromp|\breprenons\b|'
    # Portuguese
    r'\bespera\b|\bna verdade\b|\bnão,?\s+espera\b|'
    r'\bcometi um erro\b',
    re.IGNORECASE
)

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

def get_k(correct_rate_lang, total_consensus):
    if correct_rate_lang >= 0.75 * N_ANSWERS and total_consensus >= 0.75:
        return 1
    elif correct_rate_lang >= 0.5 * N_ANSWERS or total_consensus >= 0.5:
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


def dataset_priorities(candidates, tokenizer):
    n_answers = 3*N_ANSWERS

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
        "skipped_no_correct_otherlang_answers": 0, # skipped pivot language question because none of the other languages had a correct answer 
        "skipped_no_correct_answers_ids": [], # id's of questions with no correct answers in any language
        "skipped_no_correct_answers_ids_langs": [], # id-lang pairs of questions with no correct answers in that language
        "english_pivot": 0, # number of times English was selected as pivot
        "french_pivot": 0, # number of times French was selected as pivot
        "portuguese_pivot": 0 # number of times Portuguese was selected as pivot
    }

    print("Calculating answer scores...")
    for c_i in range(0, len(candidates), 3):
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
            
            for j in range(N_ANSWERS):
                # get the answer
                answer = lang_answers["candidates"][j]
                
                # scores for that answer
                scores_lang[num_id][j] = dict()
                
                # get accuracy
                acc = acc_compute_score(answer, lang_answers["ground_truth"])
                scores_lang[num_id][j]["acc"] = acc
                
                # get language consistency
                lc = is_language_consistent(answer, lang)
                scores_lang[num_id][j]["lc"] = lc
                
                # get format rewards
                fmt = format_reward(answer)
                scores_lang[num_id][j]["format"] = fmt
                
                # get backtrack count
                scores_lang[num_id][j]["bt"] = len(BACKTRACK_SIGNALS.findall(answer))
                
                if acc == 1:
                    acc_rate[num_id][lang] += 1
                    if lc == 1:
                        correct_rate[num_id][lang] += 1
    
    # iterate through questions to calculate pivot and pairwise scores     
    print("Selecting correct and incorrect answers")  
    for id in en_scores:      
        # total consensus score for the question (combining three languages)
        total_consensus[id] = sum([correct_rate[id][lang] for lang in langs])/n_answers
        
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
            for i in range(N_ANSWERS):
                # scores for that answer
                answer_scores = scores[lang][id][i]
                if answer_scores["acc"] == 1 and answer_scores["lc"] == 1: 
                    # correct answer (don't ask for correct format - too strict)
                    chosen_candidates[lang][id].append(i)
                    stats["chosen_correct"] += 1
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

    for id in en_scores:
        # skip questions that were skipped in scoring
        if id not in total_consensus:
            continue        

        entry = {
            "num_id": id,
            "iteration": ITERATION_NUMBER,

            # overall difficulty
            "total_consensus": round(total_consensus[id], 4),
            "difficulty": _difficulty_label(total_consensus[id]),

            # per-language correct rate (acc + lc) out of N_ANSWERS
            "correct_rate_en": round(correct_rate[id]["en"] / N_ANSWERS, 4),
            "correct_rate_fr": round(correct_rate[id]["fr"] / N_ANSWERS, 4),
            "correct_rate_pt": round(correct_rate[id]["pt"] / N_ANSWERS, 4),

            # per-language accuracy rate (acc only, ignoring lc) out of N_ANSWERS
            "acc_rate_en": round(acc_rate[id]["en"] / N_ANSWERS, 4),
            "acc_rate_fr": round(acc_rate[id]["fr"] / N_ANSWERS, 4),
            "acc_rate_pt": round(acc_rate[id]["pt"] / N_ANSWERS, 4),

            # which language was strongest for this question
            "pivot_lang": pivot_langs[id],

            # whether pairs were actually generated for this question
            # filled in after third loop — placeholder for now
            "pairs_generated": False,
        }
        difficulty_log.append(entry)

    # iterate to generate the final pairs
    with open(DATASET_FILE, "w", encoding="utf-8") as f:
        print("Generating pairs...")
        for candidate in candidates:
            id = candidate["numerical_id"]
            lang = candidate["language"]
            
            # scores for that question
            if id not in scores[lang]:
                # no answer for that question in that language - dataset error
                continue

            question_scores = scores[lang][id]

            if id not in total_consensus or total_consensus[id] == 0:
                # no correct answers for that question
                continue

            if not chosen_candidates[lang][id] or not rejected_candidates[lang][id]:
                # no correct or incorrect answers - useless for training
                continue
            
            if lang not in chrF_alignment[id]:
                # pivot language when no other languages found a correct answer
                continue
                
            # dynamically select number of pairs to generate per question and language
            k = get_k(correct_rate[id][lang], total_consensus[id])
            
            # sort chosen candidates by chrF alignment score descending
            def candidate_score(c_id):
                chrf_val = chrF_alignment[id][lang][c_id]
                # penalize answers with more baacktracking
                bt_count = question_scores[c_id]["bt"]
                # reward answers with correct format
                fmt = question_scores[c_id]["format"]
                return chrf_val - WEIGHT_BACKTRACK * bt_count + WEIGHT_FORMAT * fmt

            sorted_chosen = sorted(
                chrF_alignment[id][lang].keys(),
                key=candidate_score,
                reverse=True
            )
                
            # take top k, but don't exceed available candidates
            top_k_chosen = sorted_chosen[:min(k, len(sorted_chosen))]
                
            # rejected candidates
            rejected_cands = rejected_candidates[lang][id]
            # answers with wrong result but correct language consistency
            rejected_wrong_ans = [i for i in rejected_cands if question_scores[i]["acc"]==0 and question_scores[i]["lc"]==1] 
            # answers with correct result but incorrect language consistency
            rejected_inconsistent_ans = [i for i in rejected_cands if question_scores[i]["acc"]==1 and question_scores[i]["lc"]==0] 
            # other rejected ones
            other_rejected =  [i for i in rejected_cands if question_scores[i]["acc"]==0 and question_scores[i]["lc"]==0] 
                
            # rejected answers
            rejected_selected = select_rejected(
                lang,
                rejected_wrong_ans,
                rejected_inconsistent_ans,
                other_rejected,
                k
            )
                
            for chosen, rejected in zip(top_k_chosen, rejected_selected):
                # construct pairs
                # prompt
                prompt_text = LANG_TO_INSTRUCTIONS[lang].format(question=candidate["question"])
                messages = [{"role": "user", "content": prompt_text}]
                prompt_formatted = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=True,
                )
                
                pair_dict = {
                    "prompt": prompt_formatted,
                    "chosen": candidate["candidates"][chosen],
                    "rejected": candidate["candidates"][rejected] 
                }
                dpo_pairs.append(pair_dict)
                ids_with_pairs.add(id)
                stats["pairs_created"] += 1
                f.write(json.dumps(pair_dict, ensure_ascii=False) + "\n")

            stats["question_lang_with_pairs"] += 1
    
    print("Finishing difficulty log...")
    # prompts with pairs
    for entry in difficulty_log:
        # change to true if entry["num_id"] in ids_with_pairs
        if entry["num_id"] in ids_with_pairs:
            entry["pairs_generated"] = True

    return dpo_pairs, stats, difficulty_log
 
    
def main():
    #try:
        counter = 0
        candidates = []
        # load answers file
        with open(CANDIDATES_FILE, "r", encoding="utf-8") as f:
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
        
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)           
        dpo_pairs, stats, difficulty_log = dataset_priorities(candidates, tokenizer)
        
        with open(STATS_FILE, "w") as f:
            json.dump(stats, f)
        
        with open(DIFFICULTY_LOG_FILE, "w", encoding="utf-8") as f:
            for entry in difficulty_log:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        

    #except Exception as e:
    #    print(counter)
    #    print(f"Errorç: {e}")
    #    return      

if __name__ == "__main__":
    main()