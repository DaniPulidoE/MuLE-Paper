import json, argparse
from numpy import mean

parser = argparse.ArgumentParser()

parser.add_argument("--model_name", type=str, default="")
parser.add_argument("--langs", type=str, nargs="+", default=["ja", "ko", "fr", "pt", "th", "en", "es", "ar", "vi", "zh"],
                     help="Languages to include, in order. Determines the ID/OOD split (first 5 vs rest).")

args = parser.parse_args()


path = f"logs-eval/PolyMath-temp_0.9/{args.model_name}/score-eval.jsonl"


data = {}
with open(f"{path}", "r", encoding="utf-8") as files:
    for line in files:
        item = json.loads(line)
        key = list(item.keys())[0]
        data[key] = item[key]

print(len(data))

langs = args.langs

acc_langs = []
strict_acc_langs = []
think_cons_langs = []
answer_cons_langs = []
cons_langs = []
for lang in langs:
    acc_avg = 0
    strict_acc_avg = 0
    think_cons_avg = 0
    answer_cons_avg = 0
    cons_avg = 0
    for i in range(4):
        benchmark_weighted_acc = round(sum([(2 ** _i) * data[f"{lang}-{level}-{i}"][f"acc"] for _i, level in enumerate(["low", "medium", "high", "top"])]) / 15, 2)
        benchmark_weighted_strict_acc = round(sum([(2 ** _i) * data[f"{lang}-{level}-{i}"][f"strict_acc"] for _i, level in enumerate(["low", "medium", "high", "top"])]) / 15, 2)
        benchmark_thinking_lang_cons = round(sum([ data[f"{lang}-{level}-{i}"][f"thinking_lang_cons"] for level in ["low", "medium", "high", "top"]]) / 4, 2)
        benchmark_answer_lang_cons = round(sum([ data[f"{lang}-{level}-{i}"][f"answer_lang_cons"] for level in ["low", "medium", "high", "top"]]) / 4, 2)
        benchmark_lang_cons = round(sum([ data[f"{lang}-{level}-{i}"][f"all_lang_cons"] for level in ["low", "medium", "high", "top"]]) / 4, 2)


        # print(f"{lang}-{i}", benchmark_weighted_acc, benchmark_weighted_strict_acc, benchmark_thinking_lang_cons, benchmark_answer_lang_cons, benchmark_lang_cons)
        acc_avg += benchmark_weighted_acc
        strict_acc_avg += benchmark_weighted_strict_acc
        think_cons_avg += benchmark_thinking_lang_cons
        answer_cons_avg += benchmark_answer_lang_cons
        cons_avg += benchmark_lang_cons
    
    acc_langs.append(str(round(acc_avg/4, 2)))
    strict_acc_langs.append(str(round(strict_acc_avg/4, 2)))
    think_cons_langs.append(str(round(think_cons_avg/4, 2)))
    answer_cons_langs.append(str(round(answer_cons_avg/4, 2)))
    cons_langs.append(str(round(cons_avg/4, 2)))


float_strict_acc_langs = [float(i) for i in strict_acc_langs]
float_acc_langs = [float(i) for i in acc_langs]
float_cons_langs = [float(i) for i in cons_langs]


# The ID/OOD split (languages seen vs. not seen during training) only applies to the
# canonical 10-language ordering; for a custom subset we just report the overall average.
has_id_ood_split = langs == ["ja", "ko", "fr", "pt", "th", "en", "es", "ar", "vi", "zh"]

print(path)
if has_id_ood_split:
    print("Metrics", "\t".join(langs), "\tID-avg\tOOD-avg\tALL-avg")
    print("LC&Acc:\t", "\t".join(strict_acc_langs), "\t", round(mean(float_strict_acc_langs[:5]), 2), "\t", round(mean(float_strict_acc_langs[5:]), 2), "\t", round(mean(float_strict_acc_langs), 2))
    print("Acc:\t", "\t".join(acc_langs), "\t", round(mean(float_acc_langs[:5]), 2), "\t", round(mean(float_acc_langs[5:]), 2), "\t", round(mean(float_acc_langs), 2))
    print("LC:\t", "\t".join(cons_langs), "\t", round(mean(float_cons_langs[:5]), 2), "\t", round(mean(float_cons_langs[5:]), 2), "\t", round(mean(float_cons_langs), 2))
else:
    print("Metrics", "\t".join(langs), "\tALL-avg")
    print("LC&Acc:\t", "\t".join(strict_acc_langs), "\t", round(mean(float_strict_acc_langs), 2))
    print("Acc:\t", "\t".join(acc_langs), "\t", round(mean(float_acc_langs), 2))
    print("LC:\t", "\t".join(cons_langs), "\t", round(mean(float_cons_langs), 2))

print("-"*100)

float_think_cons_langs = [float(i) for i in think_cons_langs]
float_answer_cons_langs = [float(i) for i in answer_cons_langs]

# print("\t".join(think_cons_langs), "\t", round(mean(float_think_cons_langs[:5]), 2), "\t", round(mean(float_think_cons_langs[5:]), 2), "\t", round(mean(float_think_cons_langs), 2))
# print("\t".join(answer_cons_langs), "\t", round(mean(float_answer_cons_langs[:5]), 2), "\t", round(mean(float_answer_cons_langs[5:]), 2), "\t", round(mean(float_answer_cons_langs), 2))

print()
print()

# --- Per-difficulty-level breakdown (unweighted, averaged over the 4 sampled runs) ---
print("="*100)
print("Per-level breakdown (each cell averaged over cnt=0..3 sampled runs)")
print("="*100)

level_order = ["low", "medium", "high", "top"]
level_metrics = ["acc", "strict_acc", "thinking_lang_cons", "answer_lang_cons", "all_lang_cons"]

for level in level_order:
    print(f"\n--- Level: {level} ---")
    print("Metrics\t" + "\t".join(langs) + "\tLevel-avg")
    for metric in level_metrics:
        row = [round(mean([data[f"{lang}-{level}-{i}"][metric] for i in range(4)]), 2) for lang in langs]
        print(f"{metric}:\t" + "\t".join(str(v) for v in row) + f"\t{round(mean(row), 2)}")

print("\n--- Level averages across all languages ---")
print("Level\t" + "\t".join(level_metrics))
for level in level_order:
    row = [round(mean([mean([data[f"{lang}-{level}-{i}"][metric] for i in range(4)]) for lang in langs]), 2) for metric in level_metrics]
    print(f"{level}\t" + "\t".join(str(v) for v in row))

print()




