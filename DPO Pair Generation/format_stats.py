"""Decompose format_reward over iter1_questions.jsonl, per language.

Reports each format condition separately so the SUMMARY_LENGTH decision is
data-driven rather than guessed:

  open        <think> present                       (condition 1)
  close       </think> present                      (condition 2a)
  oneClose    exactly one </think>                  (what the eval's split needs)
  boxedAfter  \\boxed{} after the last </think>      (condition 2b)
  fmt800/2000/4000  full format_reward at different SUMMARY_LENGTH values
  evalFmt     what compute_score_acc_lc actually requires before scoring:
              exactly one </think>, len > 1000, boxed extractable after it

All rates are shown for every answer and conditioned on acc == 1 (the chosen
pool), plus percentiles of the gap between </think> and the first \\boxed{}.

Usage (from repo root):
  python "DPO Pair Generation/format_stats.py"
"""
import json
import re
import os
import sys
import argparse
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rewards import acc_compute_score

CANDIDATES_FILE = "./Datos/iter1_questions.jsonl"
SUMMARY_LENGTHS = [800, 2000, 4000]
LANGS = ["en", "fr", "pt"]


def analyze(text, gt):
    flags = {}
    flags["open"] = "<think>" in text
    n_close = text.count("</think>")
    flags["close"] = n_close >= 1
    flags["oneClose"] = n_close == 1

    gap = None
    boxed_after = False
    if flags["close"]:
        close_pos = text.rfind("</think>")
        boxes_after = [m.start() for m in re.finditer(r"\\boxed\{", text)
                       if m.start() > close_pos]
        if boxes_after:
            boxed_after = True
            gap = min(boxes_after) - close_pos
    flags["boxedAfter"] = boxed_after

    for L in SUMMARY_LENGTHS:
        flags[f"fmt{L}"] = (flags["open"] and flags["close"]
                            and boxed_after and gap is not None and gap <= L)

    flags["evalFmt"] = flags["oneClose"] and len(text) > 1000 and boxed_after
    flags["acc"] = acc_compute_score(text, gt) == 1.0
    return flags, gap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates_file", type=str, default=CANDIDATES_FILE)
    args = parser.parse_args()

    keys = ["open", "close", "oneClose", "boxedAfter",
            "fmt800", "fmt2000", "fmt4000", "evalFmt", "acc"]
    totals = {l: defaultdict(int) for l in LANGS}
    cond_acc = {l: defaultdict(int) for l in LANGS}
    gaps = {l: [] for l in LANGS}

    n_rec = 0
    with open(args.candidates_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            lang = rec.get("language")
            if lang not in LANGS:
                continue
            n_rec += 1
            if n_rec % 1000 == 0:
                print(f"  {n_rec} records...")
            gt = rec["ground_truth"]
            for cand in rec["candidates"]:
                flags, gap = analyze(cand, gt)
                T = totals[lang]
                T["n"] += 1
                for k in keys:
                    T[k] += flags[k]
                if flags["acc"]:
                    A = cond_acc[lang]
                    A["n"] += 1
                    for k in keys:
                        A[k] += flags[k]
                if gap is not None:
                    gaps[lang].append(gap)

    def pct(a, b):
        return f"{100 * a / b:5.1f}" if b else "  n/a"

    def table(title, data):
        print(f"\n=== {title} ===")
        header = f"{'lang':<6}{'n':>8}" + "".join(f"{k:>12}" for k in keys)
        print(header)
        print("-" * len(header))
        for l in LANGS:
            D = data[l]
            print(f"{l:<6}{D['n']:>8}" + "".join(f"{pct(D[k], D['n']):>12}" for k in keys))

    table("All answers (%)", totals)
    table("Answers with acc == 1 only (%)  <- the chosen pool", cond_acc)

    print("\n=== Gap between </think> and first \\boxed{} (chars, answers having both) ===")
    for l in LANGS:
        g = sorted(gaps[l])
        n = len(g)
        if not n:
            continue
        q = lambda p: g[min(int(p * n), n - 1)]
        over = {L: sum(1 for x in g if x > L) for L in SUMMARY_LENGTHS}
        print(f"{l}: n={n}  p50={q(0.5)}  p90={q(0.9)}  p99={q(0.99)}  max={g[-1]}  "
              + "  ".join(f">{L}: {pct(over[L], n)}%" for L in SUMMARY_LENGTHS))


if __name__ == "__main__":
    main()
