import argparse
import json
import os
import re
import statistics
from langdetect import detect_langs, DetectorFactory
from transformers import AutoTokenizer
from numpy import mean


DEFAULT_LANGS = ["ja", "ko", "fr", "pt", "th", "en", "es", "ar", "vi", "zh"]


def normalize_lang_code(code):
    if code is None:
        return ""
    c = str(code).lower()
    if c.startswith("__label__"):
        c = c[len("__label__"):]

    # Normalize common variants used by different detectors
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
        # Remove chat/control tags that can confuse language identification.
        cleaned = re.sub(r'<[^>\n]{1,200}>', ' ', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned

    def _detect_langdetect(self, text, target_lang):
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
            # fasttext.predict rejects any whitespace that tokenizes oddly, flatten all of it
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
        re_pred = self._strip_math(pred)
        if len(re_pred) <= 15:
            return True

        target_lang = normalize_lang_code(language)
        if self.backend == "fasttext":
            return self._detect_fasttext(re_pred, target_lang)
        return self._detect_langdetect(re_pred, target_lang)



parser = argparse.ArgumentParser()
parser.add_argument("--res_path", type=str, default="")
parser.add_argument("--tokenizer_path", type=str, default=None)
parser.add_argument("--trust_remote_code", action="store_true")
parser.add_argument("--langs", type=str, nargs="+", default=DEFAULT_LANGS)
parser.add_argument("--num_samples", type=int, default=None)
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
    help="Report per-language variance/std across sample indices (prediction_0..k).",
)

args = parser.parse_args()
    
path = args.res_path
langs = args.langs
detector = LanguageConsistencyDetector(
    backend=args.lang_detector,
    fasttext_model_path=args.fasttext_model_path,
    fasttext_min_prob=args.fasttext_min_prob,
)
print(
    f"[LC detector] backend={args.lang_detector}"
    + (
        f" model={args.fasttext_model_path} min_prob={args.fasttext_min_prob}"
        if args.lang_detector == "fasttext"
        else ""
    )
)

acc_langs = []
strict_acc_langs = []
think_cons_langs = []
answer_cons_langs = []
cons_langs = []
response_length_langs = []
per_lang_sample_stats = {}

tokenizer = None
if args.tokenizer_path is not None and args.tokenizer_path != "":
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path,
        trust_remote_code=args.trust_remote_code
    )


for lang in langs:
    data = json.load(open(f"{path}/{lang}.json"))
    if len(data) == 0:
        continue

    if args.num_samples is not None:
        sample_indices = list(range(args.num_samples))
    else:
        sample_indices = sorted(
            int(k.rsplit("_", 1)[1]) for k in data[0].keys() if k.startswith("prediction_")
        )
        if len(sample_indices) == 0:
            sample_indices = [0]

    sources = ["AIME2024", "AIME2025", "CNMO", "MATH500"]
    dif_cnt_acc = 0
    dif_cnt_strict_acc = 0
    dif_cnt_think_cons = 0
    dif_cnt_answer_cons = 0
    dif_cnt_cons = 0

    response_length_list = []

    for i_cnt in sample_indices:
        acc_dict = {}
        strict_acc_dict = {}
        think_cons_dict = {}
        answer_cons_dict = {}
        cons_dict = {}

        for i_source in sources:
            acc_dict[i_source] = {"num": 0, "acc": 0}
            strict_acc_dict[i_source] = {"num": 0, "acc&cons": 0}
            think_cons_dict[i_source] = {"num": 0, "think_cons": 0}
            answer_cons_dict[i_source] = {"num": 0, "answer_cons": 0}
            cons_dict[i_source] = {"num": 0, "cons": 0}


        for item in data:
            source_name = item.get("data_source", item.get("source", "MATH500"))
            if source_name not in sources:
                source_name = "MATH500"
            acc_dict[source_name]["num"] += 1
            strict_acc_dict[source_name]["num"] += 1
            think_cons_dict[source_name]["num"] += 1
            answer_cons_dict[source_name]["num"] += 1
            cons_dict[source_name]["num"] += 1

            if item[f"correct_{i_cnt}"]: 
                acc_dict[source_name]["acc"] += 1
            
            think_pred = None
            answer_pred = None
            generated_text = item[f"prediction_{i_cnt}"]
            if tokenizer is not None:
                response_length_list.append(len(tokenizer(generated_text)['input_ids']))
            if "</think>" in generated_text:
                # Split once so extra tags in the answer do not invalidate LC parsing.
                think_pred, answer_pred = generated_text.split("</think>", 1)

            if think_pred is not None:
                whether_cons_think = detector.is_consistent(think_pred, lang)
            else:
                whether_cons_think = False

            if answer_pred is not None:
                whether_cons_answer = detector.is_consistent(answer_pred, lang)
            else:
                whether_cons_answer = False

            if whether_cons_think:
                think_cons_dict[source_name]["think_cons"] += 1

            if whether_cons_answer:
                answer_cons_dict[source_name]["answer_cons"] += 1
             
            if whether_cons_think and whether_cons_answer:
                cons_dict[source_name]["cons"] += 1

            if whether_cons_think and whether_cons_answer and item[f"correct_{i_cnt}"]: 
                strict_acc_dict[source_name]["acc&cons"] += 1

        dif_source_acc = 0
        dif_source_strict_acc = 0
        dif_source_think_cons = 0
        dif_source_answer_cons = 0
        dif_source_cons = 0
        present_source_cnt = 0

        for i_source in sources:
            if acc_dict[i_source]["num"] == 0:
                continue
            present_source_cnt += 1
            dif_source_acc += acc_dict[i_source]["acc"] / acc_dict[i_source]["num"] * 100
            dif_source_strict_acc += strict_acc_dict[i_source]["acc&cons"] / strict_acc_dict[i_source]["num"] * 100
            dif_source_think_cons += think_cons_dict[i_source]["think_cons"] / think_cons_dict[i_source]["num"] * 100
            dif_source_answer_cons += answer_cons_dict[i_source]["answer_cons"] / answer_cons_dict[i_source]["num"] * 100
            dif_source_cons += cons_dict[i_source]["cons"] / cons_dict[i_source]["num"] * 100

        if present_source_cnt == 0:
            continue

        # print(lang, acc_dict, round(dif_source_acc/4, 4))
        # print(lang, strict_acc_dict, round(dif_source_strict_acc/4, 4))
        # print(lang, think_cons_dict, round(dif_source_think_cons/4, 4))
        # print(lang, answer_cons_dict, round(dif_source_answer_cons/4, 4))
        # print(lang, cons_dict, round(dif_source_cons/4, 4))

        dif_cnt_acc += dif_source_acc / present_source_cnt
        dif_cnt_strict_acc += dif_source_strict_acc / present_source_cnt
        dif_cnt_think_cons += dif_source_think_cons / present_source_cnt
        dif_cnt_answer_cons += dif_source_answer_cons / present_source_cnt
        dif_cnt_cons += dif_source_cons / present_source_cnt

    per_lang_sample_stats[lang] = {
        "LC&Acc": [],
        "Acc": [],
        "LC_think": [],
        "LC_answer": [],
        "LC": [],
    }
    for i_cnt in sample_indices:
        # Recompute per-sample metric entries from aggregated counters by
        # subtracting the running means unreliable. Dtore direct entries.
        # Reconstruct from the already accumulated totals by re-looping
        # lightweight source stats for each sample index
        acc_dict = {s: {"num": 0, "acc": 0} for s in sources}
        strict_acc_dict = {s: {"num": 0, "acc&cons": 0} for s in sources}
        think_cons_dict = {s: {"num": 0, "think_cons": 0} for s in sources}
        answer_cons_dict = {s: {"num": 0, "answer_cons": 0} for s in sources}
        cons_dict = {s: {"num": 0, "cons": 0} for s in sources}

        for item in data:
            source_name = item.get("data_source", item.get("source", "MATH500"))
            if source_name not in sources:
                source_name = "MATH500"
            acc_dict[source_name]["num"] += 1
            strict_acc_dict[source_name]["num"] += 1
            think_cons_dict[source_name]["num"] += 1
            answer_cons_dict[source_name]["num"] += 1
            cons_dict[source_name]["num"] += 1

            if item[f"correct_{i_cnt}"]:
                acc_dict[source_name]["acc"] += 1

            generated_text = item[f"prediction_{i_cnt}"]
            think_pred = None
            answer_pred = None
            if "</think>" in generated_text:
                think_pred, answer_pred = generated_text.split("</think>", 1)

            whether_cons_think = detector.is_consistent(think_pred, lang) if think_pred is not None else False
            whether_cons_answer = detector.is_consistent(answer_pred, lang) if answer_pred is not None else False

            if whether_cons_think:
                think_cons_dict[source_name]["think_cons"] += 1
            if whether_cons_answer:
                answer_cons_dict[source_name]["answer_cons"] += 1
            if whether_cons_think and whether_cons_answer:
                cons_dict[source_name]["cons"] += 1
            if whether_cons_think and whether_cons_answer and item[f"correct_{i_cnt}"]:
                strict_acc_dict[source_name]["acc&cons"] += 1

        present_source_cnt = 0
        sample_acc = 0.0
        sample_strict = 0.0
        sample_think = 0.0
        sample_answer = 0.0
        sample_cons = 0.0
        for i_source in sources:
            if acc_dict[i_source]["num"] == 0:
                continue
            present_source_cnt += 1
            sample_acc += acc_dict[i_source]["acc"] / acc_dict[i_source]["num"] * 100
            sample_strict += strict_acc_dict[i_source]["acc&cons"] / strict_acc_dict[i_source]["num"] * 100
            sample_think += think_cons_dict[i_source]["think_cons"] / think_cons_dict[i_source]["num"] * 100
            sample_answer += answer_cons_dict[i_source]["answer_cons"] / answer_cons_dict[i_source]["num"] * 100
            sample_cons += cons_dict[i_source]["cons"] / cons_dict[i_source]["num"] * 100

        if present_source_cnt == 0:
            continue
        per_lang_sample_stats[lang]["Acc"].append(sample_acc / present_source_cnt)
        per_lang_sample_stats[lang]["LC&Acc"].append(sample_strict / present_source_cnt)
        per_lang_sample_stats[lang]["LC_think"].append(sample_think / present_source_cnt)
        per_lang_sample_stats[lang]["LC_answer"].append(sample_answer / present_source_cnt)
        per_lang_sample_stats[lang]["LC"].append(sample_cons / present_source_cnt)

    sample_cnt = max(len(sample_indices), 1)
    acc_langs.append(str(round(dif_cnt_acc/sample_cnt, 4)))
    strict_acc_langs.append(str(round(dif_cnt_strict_acc/sample_cnt, 4)))
    think_cons_langs.append(str(round(dif_cnt_think_cons/sample_cnt, 4)))
    answer_cons_langs.append(str(round(dif_cnt_answer_cons/sample_cnt, 4)))
    cons_langs.append(str(round(dif_cnt_cons/sample_cnt, 4)))

    if tokenizer is not None and len(response_length_list) > 0:
        response_length_langs.append(str(round(mean(response_length_list), 4)))


print(path)

float_strict_acc_langs = [float(i) for i in strict_acc_langs]
float_acc_langs = [float(i) for i in acc_langs]
float_cons_langs = [float(i) for i in cons_langs]
float_response_length_langs = [float(i) for i in response_length_langs]

if langs == DEFAULT_LANGS:
    print("Metrics", "ja", "ko", "fr", "pt", "th", "en", "es", "ar", "vi", "zh", "ID-avg", "OOD-avg", "ALL-avg")
    print(
        "LC&Acc:\t", "\t".join(strict_acc_langs), "\t",
        round(mean(float_strict_acc_langs[:5]), 2), "\t",
        round(mean(float_strict_acc_langs[5:]), 2), "\t",
        round(mean(float_strict_acc_langs), 2)
    )
    print(
        "Acc:\t", "\t".join(acc_langs), "\t",
        round(mean(float_acc_langs[:5]), 2), "\t",
        round(mean(float_acc_langs[5:]), 2), "\t",
        round(mean(float_acc_langs), 2)
    )
    print(
        "LC:\t", "\t".join(cons_langs), "\t",
        round(mean(float_cons_langs[:5]), 2), "\t",
        round(mean(float_cons_langs[5:]), 2), "\t",
        round(mean(float_cons_langs), 2)
    )
else:
    print("Metrics", *langs, "ALL-avg")
    print("LC&Acc:\t", "\t".join(strict_acc_langs), "\t", round(mean(float_strict_acc_langs), 2))
    print("Acc:\t", "\t".join(acc_langs), "\t", round(mean(float_acc_langs), 2))
    print("LC:\t", "\t".join(cons_langs), "\t", round(mean(float_cons_langs), 2))
# print("Response-length\t", "\t".join(response_length_langs), "\t", round(mean(float_response_length_langs[:5]), 2), "\t", round(mean(float_response_length_langs[5:]), 2), "\t", round(mean(float_response_length_langs), 2))

print("-"*100)

float_think_cons_langs = [float(i) for i in think_cons_langs]
float_answer_cons_langs = [float(i) for i in answer_cons_langs]

# print("\t".join(think_cons_langs), "\t", round(mean(float_think_cons_langs[:5]), 2), "\t", round(mean(float_think_cons_langs[5:]), 2), "\t", round(mean(float_think_cons_langs), 2))
# print("\t".join(answer_cons_langs), "\t", round(mean(float_answer_cons_langs[:5]), 2), "\t", round(mean(float_answer_cons_langs[5:]), 2), "\t", round(mean(float_answer_cons_langs), 2))

print()
print()

if args.report_variance:
    def _var_stats(values):
        if not values:
            return (0, 0.0, 0.0, 0.0)
        n = len(values)
        m = statistics.mean(values)
        var = statistics.pvariance(values) if n > 1 else 0.0
        std = statistics.pstdev(values) if n > 1 else 0.0
        return (n, m, var, std)

    print("[Summary Table] mean±std(var)")
    print("Metrics", *langs, "ALL-avg")
    for metric_name in ["LC&Acc", "Acc", "LC"]:
        cells = []
        means = []
        vars_ = []
        stds = []
        for lang in langs:
            arr = per_lang_sample_stats.get(lang, {}).get(metric_name, [])
            n, m, var, std = _var_stats(arr)
            if n > 0:
                means.append(m)
                vars_.append(var)
                stds.append(std)
            cells.append(f"{m:.2f}±{std:.2f}({var:.2f})")
        all_cell = "0.00±0.00(0.00)"
        if len(means) > 0:
            all_cell = (
                f"{statistics.mean(means):.2f}"
                f"±{statistics.mean(stds):.2f}"
                f"({statistics.mean(vars_):.2f})"
            )
        print(f"{metric_name}:\t", "\t".join(cells), "\t", all_cell)
    print("-" * 100)

    print("[Variance] per-language across sample indices")
    for lang in langs:
        if lang not in per_lang_sample_stats:
            continue
        row = per_lang_sample_stats[lang]
        print(f"{lang}:")
        for metric_name in ["LC&Acc", "Acc", "LC", "LC_think", "LC_answer"]:
            n, m, var, std = _var_stats(row.get(metric_name, []))
            print(
                f"  {metric_name}: n={n} mean={m:.4f} var={var:.4f} std={std:.4f}"
            )
    print("-" * 100)
