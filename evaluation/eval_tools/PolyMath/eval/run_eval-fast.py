import os
import re
import json
import numpy as np
import statistics
from collections import defaultdict
from tqdm import tqdm
from argparse import ArgumentParser

# from scripts import math_equal
try:
    from langdetect import detect_langs, DetectorFactory
except Exception:
    detect_langs = None
    DetectorFactory = None

language_list = ["en", "zh", "ar", "bn", "de", "es", "fr", "id", "it", "ja", "ko", "ms", "pt", "ru", "sw", "te", "th", "vi", ]
language_list = ["ko", "ja", "pt", "th", "en", "zh", "ar", "es", "fr", "vi"]
level_list = ["low", "medium", "high", "top"]


def report_variance(args):
    score_file = args.score_file or os.path.join(
        f"logs-eval/PolyMath-temp_0.9/{args.model}",
        "score-eval.jsonl"
    )
    if not os.path.exists(score_file):
        raise FileNotFoundError(f"score-eval file not found: {score_file}")

    metrics = ["strict_acc", "acc", "thinking_lang_cons", "answer_lang_cons", "all_lang_cons"]
    grouped = defaultdict(lambda: defaultdict(list))

    with open(score_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            for key, vals in rec.items():
                m = re.match(r"(.+)-([^-]+)-([0-9]+)$", key)
                if not m:
                    continue
                lang, level, _cnt = m.groups()
                group = f"{lang}-{level}"
                for metric in metrics:
                    if metric in vals:
                        grouped[group][metric].append(float(vals[metric]))

    if len(grouped) == 0:
        print(f"No parseable entries found in {score_file}")
        return

    def _stats(values):
        n = len(values)
        if n == 0:
            return (0, 0.0, 0.0, 0.0)
        mean_v = statistics.mean(values)
        var_v = statistics.pvariance(values) if n > 1 else 0.0
        std_v = statistics.pstdev(values) if n > 1 else 0.0
        return (n, mean_v, var_v, std_v)

    print(f"[Variance] source={score_file}")
    for group in sorted(grouped.keys()):
        print(group)
        for metric in metrics:
            n, mean_v, var_v, std_v = _stats(grouped[group][metric])
            print(f"  {metric}: n={n} mean={mean_v:.4f} var={var_v:.4f} std={std_v:.4f}")

    # Macro summary across groups based on per-group means.
    print("[Variance] macro across groups (means of means)")
    for metric in metrics:
        per_group_means = []
        for group in grouped.keys():
            vals = grouped[group][metric]
            if len(vals) > 0:
                per_group_means.append(statistics.mean(vals))
        n, mean_v, var_v, std_v = _stats(per_group_means)
        print(f"  {metric}: groups={n} mean={mean_v:.4f} var={var_v:.4f} std={std_v:.4f}")


def _load_score_rows(score_file):
    rows = []
    if not os.path.exists(score_file):
        raise FileNotFoundError(f"score-eval file not found: {score_file}")

    with open(score_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            for key, vals in rec.items():
                parts = key.rsplit("-", 2)
                if len(parts) != 3:
                    continue
                lang, level, cnt = parts
                rows.append({
                    "lang": lang,
                    "level": level,
                    "cnt": cnt,
                    "strict_acc": float(vals.get("strict_acc", 0.0)),
                    "acc": float(vals.get("acc", 0.0)),
                    "thinking_lang_cons": float(vals.get("thinking_lang_cons", 0.0)),
                    "answer_lang_cons": float(vals.get("answer_lang_cons", 0.0)),
                    "all_lang_cons": float(vals.get("all_lang_cons", 0.0)),
                })
    return rows


def _fmt_table(rows, headers):
    str_rows = [[str(c) for c in r] for r in rows]
    widths = [len(h) for h in headers]
    for r in str_rows:
        for i, c in enumerate(r):
            widths[i] = max(widths[i], len(c))
    header_line = " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    sep_line = "-+-".join("-" * widths[i] for i in range(len(headers)))
    body = [" | ".join(c.ljust(widths[i]) for i, c in enumerate(r)) for r in str_rows]
    return "\n".join([header_line, sep_line] + body)


def report_summary_table(args):
    score_file = args.score_file or os.path.join(
        f"logs-eval/PolyMath-temp_0.9/{args.model}",
        "score-eval.jsonl"
    )
    rows = _load_score_rows(score_file)
    if len(rows) == 0:
        print(f"No parseable entries found in {score_file}")
        return

    metrics = ["strict_acc", "acc", "thinking_lang_cons", "answer_lang_cons", "all_lang_cons"]
    by_lang = defaultdict(list)
    for r in rows:
        by_lang[r["lang"]].append(r)

    def _stats(vals, metric):
        arr = [v[metric] for v in vals]
        if len(arr) == 0:
            return (0, 0.0, 0.0, 0.0)
        m = statistics.mean(arr)
        var = statistics.pvariance(arr) if len(arr) > 1 else 0.0
        std = statistics.pstdev(arr) if len(arr) > 1 else 0.0
        return (len(arr), m, var, std)

    table_rows = []
    for lang in sorted(by_lang.keys()):
        vals = by_lang[lang]
        _, s_mean, s_var, s_std = _stats(vals, "strict_acc")
        _, a_mean, a_var, a_std = _stats(vals, "acc")
        _, t_mean, t_var, t_std = _stats(vals, "thinking_lang_cons")
        _, ans_mean, ans_var, ans_std = _stats(vals, "answer_lang_cons")
        _, all_mean, all_var, all_std = _stats(vals, "all_lang_cons")
        table_rows.append([
            lang,
            len(vals),
            f"{s_mean:.2f}±{s_std:.2f}({s_var:.2f})",
            f"{a_mean:.2f}±{a_std:.2f}({a_var:.2f})",
            f"{t_mean:.2f}±{t_std:.2f}({t_var:.2f})",
            f"{ans_mean:.2f}±{ans_std:.2f}({ans_var:.2f})",
            f"{all_mean:.2f}±{all_std:.2f}({all_var:.2f})",
        ])

    # Macro row across languages (equal language weight).
    per_lang_stats = {}
    for lang, vals in by_lang.items():
        per_lang_stats[lang] = {m: _stats(vals, m) for m in metrics}
    def _macro(metric, idx):
        # idx: 1=mean, 2=var, 3=std
        arr = [per_lang_stats[l][metric][idx] for l in per_lang_stats]
        return statistics.mean(arr) if len(arr) else 0.0
    macro_row = [
        "ALL-macro",
        len(rows),
        f"{_macro('strict_acc', 1):.2f}±{_macro('strict_acc', 3):.2f}({_macro('strict_acc', 2):.2f})",
        f"{_macro('acc', 1):.2f}±{_macro('acc', 3):.2f}({_macro('acc', 2):.2f})",
        f"{_macro('thinking_lang_cons', 1):.2f}±{_macro('thinking_lang_cons', 3):.2f}({_macro('thinking_lang_cons', 2):.2f})",
        f"{_macro('answer_lang_cons', 1):.2f}±{_macro('answer_lang_cons', 3):.2f}({_macro('answer_lang_cons', 2):.2f})",
        f"{_macro('all_lang_cons', 1):.2f}±{_macro('all_lang_cons', 3):.2f}({_macro('all_lang_cons', 2):.2f})",
    ]
    table_rows.append(macro_row)

    headers = ["lang", "n_rows", "strict_acc", "acc", "lc_think", "lc_answer", "lc_all"]
    print(f"[Summary Table] source={score_file}")
    print("cell format: mean±std(var)")
    print(_fmt_table(table_rows, headers))


def normalize_lang_code(code):
    if code is None:
        return ""
    c = str(code).lower()
    if c.startswith("__label__"):
        c = c[len("__label__"):]

    if c.startswith("zh"):
        return "zh"
    if c.startswith("pt"):
        return "pt"
    if c == "jp":
        return "ja"
    if c in {"ko", "fr", "es", "en", "ar", "vi", "th", "ja", "zh", "pt"}:
        return c
    return c


class LanguageConsistencyDetector:
    def __init__(self, backend="langdetect", fasttext_model_path="eval_tools/langid/lid.176.ftz", fasttext_min_prob=0.7):
        self.backend = backend
        self.fasttext_min_prob = fasttext_min_prob
        self.fasttext_model = None
        self.fasttext_model_path = fasttext_model_path

        if self.backend == "fasttext":
            try:
                import fasttext  # type: ignore
            except Exception as exc:
                raise RuntimeError(
                    "FastText backend selected but fasttext is unavailable. "
                    "Install with: python -m pip install fasttext-wheel"
                ) from exc

            try:
                from fasttext import FastText as _ft_mod  # type: ignore
                if not getattr(_ft_mod, "_np2_shim_applied", False):
                    _real_np_array = _ft_mod.np.array

                    def _np_array_shim(*a, **kw):
                        if kw.get("copy", True) is False:
                            try:
                                return _real_np_array(*a, **{**kw, "copy": None})
                            except TypeError:
                                kw.pop("copy", None)
                                return _real_np_array(*a, **kw)
                        return _real_np_array(*a, **kw)

                    _ft_mod.np.array = _np_array_shim
                    _ft_mod._np2_shim_applied = True
            except Exception as _shim_exc:
                print(f"[LC detector] numpy-2 shim not applied: {_shim_exc}")

            if not os.path.exists(self.fasttext_model_path):
                raise FileNotFoundError(
                    f"FastText model not found at '{self.fasttext_model_path}'. "
                    "Download lid.176.ftz (https://fasttext.cc/docs/en/language-identification.html)."
                )
            self.fasttext_model = fasttext.load_model(self.fasttext_model_path)

    @staticmethod
    def _strip_math(text):
        cleaned = re.sub(r'\$\$.*?\$\$|\\\(.*?\\\)|\\\[.*?\\\]', '', text, flags=re.DOTALL)
        cleaned = re.sub(r'\$[^$]*\$', '', cleaned)
        cleaned = re.sub(r'<[^>\n]{1,200}>', ' ', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned

    def _detect_langdetect(self, text, target_lang):
        if detect_langs is None or DetectorFactory is None:
            raise RuntimeError(
                "langdetect backend selected but langdetect is unavailable. "
                "Install with: python -m pip install langdetect"
            )
        try:
            DetectorFactory.seed = 42
            lang_prob = detect_langs(text)
            if len(lang_prob) != 1:
                return False
            pred_lang = normalize_lang_code(lang_prob[0].lang)
            return pred_lang == target_lang
        except Exception:
            return False

    _ft_err_reported = False

    def _detect_fasttext(self, text, target_lang):
        try:
            clean = re.sub(r'\s+', ' ', text).strip()
            if not clean:
                return False
            labels, probs = self.fasttext_model.predict(clean, k=1)
            if len(labels) == 0:
                return False
            pred_lang = normalize_lang_code(labels[0])
            prob = float(probs[0]) if len(probs) else 0.0
            return pred_lang == target_lang and prob >= self.fasttext_min_prob
        except Exception as exc:
            if not LanguageConsistencyDetector._ft_err_reported:
                import traceback
                print(f"[LC detector] fasttext predict failed: {type(exc).__name__}: {exc}")
                traceback.print_exc()
                print(f"[LC detector] offending text (first 200 chars): {text[:200]!r}")
                LanguageConsistencyDetector._ft_err_reported = True
            return False

    def is_consistent(self, pred, language):
        if pred is None:
            return False
        cleaned = self._strip_math(pred)
        if len(cleaned) <= 15:
            return True

        target_lang = normalize_lang_code(language)
        if self.backend == "fasttext":
            return self._detect_fasttext(cleaned, target_lang)
        return self._detect_langdetect(cleaned, target_lang)


def extract_boxed_content(text):
    pattern = re.compile(r'boxed{')
    text = text.replace(' ', '')

    matches = pattern.finditer(text)
    results = []
    for match in matches:
        start_pos = match.end()
        brace_count = 1
        i = start_pos
        while i < len(text) and brace_count > 0:
            if text[i] == '{':
                brace_count += 1
            elif text[i] == '}':
                brace_count -= 1
            i += 1
        if brace_count == 0:
            results.append(text[start_pos:i-1])
    return results


def evaluation(args):
    from math_verify import parse, verify

    model = args.model
    language = args.language
    level = args.level
    cnt = args.cnt
    detector = LanguageConsistencyDetector(
        backend=args.lang_detector,
        fasttext_model_path=args.fasttext_model_path,
        fasttext_min_prob=args.fasttext_min_prob,
    )
    if cnt == "0":
        print(
            f"[LC detector] backend={args.lang_detector}"
            + (
                f" model={args.fasttext_model_path} min_prob={args.fasttext_min_prob}"
                if args.lang_detector == "fasttext"
                else ""
            )
        )
    print(model, language, level, cnt)

    output_file = f"logs-eval/PolyMath-temp_0.9/{model}/{level}/{language}.json"
    try:
        data = json.load(open(output_file, 'r', encoding='utf-8'))
        
    except FileNotFoundError:
        ori_data = {}
        for seed in [10, 20, 30, 40]:
            ori_data[seed] = []
            output_file = output_file.replace(f"{language}.json", f"{language}-seed_{seed}.jsonl")
            with open(output_file, 'r', encoding='utf-8') as file:
                for line in file:
                    item = json.loads(line)
                    ori_data[seed].append(item)
            assert len(ori_data[seed]) == 125
        data = []
        for i in range(125):
            data.append({
                "id": ori_data[10][i]["id"],
                "question": ori_data[10][i]["question"],
                "answer": ori_data[10][i]["answer"],
                "thinking_pred_0": ori_data[10][i]["thinking_pred"],
                "answer_pred_0": ori_data[10][i]["answer_pred"],
                "thinking_pred_1": ori_data[20][i]["thinking_pred"],
                "answer_pred_1": ori_data[20][i]["answer_pred"],
                "thinking_pred_2": ori_data[30][i]["thinking_pred"],
                "answer_pred_2": ori_data[30][i]["answer_pred"],
                "thinking_pred_3": ori_data[40][i]["thinking_pred"],
                "answer_pred_3": ori_data[40][i]["answer_pred"],
            })

    if len(data) == 0:
        print("Warning! Test data is empty for this language/level/cnt; skipping.")
        return
    elif len(data) < 125:
        print(f"Warning! Test data is incomplete, current data size: {len(data)}")
    elif len(data) > 125:
        print(f"Warning! Test data is redundant, current data size: {len(data)}")
    else:
        pass


    acc, strict_acc, thinking_lang_cons, answer_lang_cons, all_lang_cons = 0, 0, 0, 0, 0
    for i in range(len(data)):
        # idx = data[i]["idx"]
        # question = data[i]["question"]
        answer = data[i]["answer"]
        thinking_pred = data[i][f"thinking_pred_{cnt}"]
        answer_pred = data[i][f"answer_pred_{cnt}"]

        ### answer extraction & correctness judgement
        try:
            extracted_pred = extract_boxed_content(answer_pred)
            extracted_pred = extracted_pred[0] if len(extracted_pred) > 0 else None
            # acc_binary = math_equal(extracted_pred, answer)
            if extracted_pred is not None:
                gold = parse('$' + answer + '$')
                pred = parse('$' + extracted_pred + '$')
                acc_binary = verify(gold, pred)
            else:
                acc_binary = False
        except:
            acc_binary = False
        acc += 1 if acc_binary else 0
        
        
        ### language consistency judgement
        thinking_lang_cons_binary = detector.is_consistent(thinking_pred, language)
        answer_lang_cons_binary = detector.is_consistent(answer_pred, language)
        
        thinking_lang_cons += 1 if thinking_lang_cons_binary else 0
        answer_lang_cons += 1 if answer_lang_cons_binary else 0
        all_lang_cons += 1 if thinking_lang_cons_binary and answer_lang_cons_binary else 0

        if thinking_lang_cons_binary and answer_lang_cons_binary and acc_binary:
            strict_acc += 1
    
    acc = round(acc / len(data) * 100, 2)
    strict_acc = round(strict_acc / len(data) * 100, 2)
    thinking_lang_cons = round(thinking_lang_cons / len(data) * 100, 2)
    answer_lang_cons = round(answer_lang_cons / len(data) * 100, 2)
    all_lang_cons = round(all_lang_cons / len(data) * 100, 2)

    print(f"Test Data Size: {len(data)}; {language}-{level}-{cnt}\n"
        f"Strict Accuracy (%) = {strict_acc}\n"
        f"Accuracy (%) = {acc}\n"
        f"Language Consistency (thinking) (%) = {thinking_lang_cons}\n"
        f"Language Consistency (answer) (%) = {answer_lang_cons}\n"
        f"ALl language consistency (%) = {all_lang_cons}")
    print("*"*30)

    res_dict = {}
    res_dict[f"{language}-{level}-{cnt}"] = {
        "strict_acc": strict_acc,
        "acc": acc,
        "thinking_lang_cons": thinking_lang_cons,
        "answer_lang_cons": answer_lang_cons,
        "all_lang_cons": all_lang_cons
    }

    # Save results
    score_file = os.path.join(f"logs-eval/PolyMath-temp_0.9/{model}", "score-eval.jsonl")
    save_f_jsonlines = open(score_file, 'a+', encoding="utf-8")
    save_f_jsonlines.write(json.dumps(res_dict, ensure_ascii=False) + '\n')
    save_f_jsonlines.flush()




if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--model', type=str, required=False, default=None)
    parser.add_argument('--language', type=str, required=False, default=None)
    parser.add_argument('--level', type=str, required=False, default=None)
    parser.add_argument('--cnt', type=str, required=False, default=None)
    parser.add_argument(
        "--lang_detector",
        type=str,
        choices=["langdetect", "fasttext"],
        default=os.getenv("LANG_DETECTOR", "langdetect"),
    )
    parser.add_argument(
        "--fasttext_model_path",
        type=str,
        default=os.getenv("FASTTEXT_LID_PATH", "eval_tools/langid/lid.176.ftz"),
    )
    parser.add_argument(
        "--fasttext_min_prob",
        type=float,
        default=float(os.getenv("FASTTEXT_MIN_PROB", "0.7")),
    )
    parser.add_argument(
        "--report_variance",
        action="store_true",
        help="Read score-eval.jsonl and report mean/variance/std across cnt runs.",
    )
    parser.add_argument(
        "--score_file",
        type=str,
        default=None,
        help="Optional explicit path to score-eval.jsonl for --report_variance/--report_table.",
    )
    parser.add_argument(
        "--report_table",
        action="store_true",
        help="Print a single aggregated language summary table from score-eval.jsonl.",
    )


    args = parser.parse_args()
    if args.report_variance or args.report_table:
        if args.model is None and args.score_file is None:
            parser.error("--report_variance/--report_table requires --model or --score_file.")
        if args.report_table:
            report_summary_table(args)
        if args.report_variance:
            # Keep variance output available in the same call when requested.
            report_variance(args)
    else:
        missing = []
        if args.model is None:
            missing.append("--model")
        if args.language is None:
            missing.append("--language")
        if args.level is None:
            missing.append("--level")
        if args.cnt is None:
            missing.append("--cnt")
        if len(missing) > 0:
            parser.error(f"Missing required args for evaluation mode: {' '.join(missing)}")
        evaluation(args)
