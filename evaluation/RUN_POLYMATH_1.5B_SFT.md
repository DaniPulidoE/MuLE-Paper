# PolyMath eval: XueZhang-bjtu/1.5B-cold-start-SFT (en/es/fr/pt)

Run on a machine with an NVIDIA GPU + CUDA (this pipeline uses vLLM). Branch: `polymath_evaluation`.

## 1. Setup (once)

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r evaluation/requirements-py311.txt
mkdir -p evaluation/eval_tools/langid
curl -L https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz -o evaluation/eval_tools/langid/lid.176.ftz
```

## 2. Generate responses, per difficulty level

```bash
cd evaluation
TEMP=0.9
MODEL_PATH="XueZhang-bjtu/1.5B-cold-start-SFT"
MODEL_NAME="1.5B-cold-start-SFT"

for LVL in low medium high top; do
  for L in en es fr pt; do
    python3 eval_tools/PolyMath/polymath_res_gen.py \
      --lang "$L" --level "$LVL" --temp $TEMP \
      --model_path "$MODEL_PATH" --model_name "$MODEL_NAME" \
      --save_path logs-eval/PolyMath-temp_${TEMP} \
      --gpu_memory_utilization 0.95 --max_model_len 16384 --max_tokens 14336 \
      --num_samples 4 --top_p 0.95 --max_num_seqs 128 --max_num_batched_tokens 65536
  done
done
```

Saves full raw text (thinking + answer, 4 samples each) to
`logs-eval/PolyMath-temp_0.9/1.5B-cold-start-SFT/<level>/<lang>.json` — this is what you'll
use later for backtrack/token-count analysis.

## 3. Score it

```bash
rm -f logs-eval/PolyMath-temp_${TEMP}/${MODEL_NAME}/score-eval.jsonl
for L in en es fr pt; do
  for LVL in low medium high top; do
    for CNT in 0 1 2 3; do
      python3 eval_tools/PolyMath/eval/run_eval-fast.py \
        --model "$MODEL_NAME" --language "$L" --level "$LVL" --cnt "$CNT" \
        --lang_detector fasttext --fasttext_model_path eval_tools/langid/lid.176.ftz --fasttext_min_prob 0.2
    done
  done
done
```

## 4. Per-level breakdown

```bash
python3 eval_tools/PolyMath/cal-polymath-acc.py --model_name "$MODEL_NAME" --langs en es fr pt
```

Prints the existing weighted per-language score, plus a section breaking accuracy/consistency
out separately for `low/medium/high/top` (per language and averaged across the 4 languages) —
added to `eval_tools/PolyMath/cal-polymath-acc.py` on this branch, not blended across levels
like the original output.

## 5. Backtracking / token / LC / accuracy analysis (per level)

Once the generation JSONs from step 2 exist locally (copy `logs-eval/PolyMath-temp_0.9/1.5B-cold-start-SFT/`
back from the GPU machine if generation ran elsewhere), run the analysis scripts in
`PolyMath Answers Statistics/` (sibling folder to `evaluation/`, mirrors `RL Train Answers Statistics/`):

```bash
cd "PolyMath Answers Statistics"
python3 Total_answers_stats.py    # per-level (and per-level x language) tables + plots
python3 Per_question_stats.py     # backtracks/tokens vs. per-question difficulty, by level
```

Both read `../evaluation/logs-eval/PolyMath-temp_0.9/1.5B-cold-start-SFT/<level>/<lang>.json`
(edit `RESULTS_DIR`/`LEVELS`/`LANGS` at the top of `polymath_data.py` if paths differ) and reuse
the same accuracy/backtrack/language-consistency logic as `RL Train Answers Statistics/`, so
results are directly comparable. Plots are saved to `./Plots/`.