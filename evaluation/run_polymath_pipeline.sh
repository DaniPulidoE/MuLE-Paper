#!/bin/bash
# PolyMath generation + scoring + per-level report for a single model, scoped to en/es/fr/pt.
# Run from the "evaluation" directory, inside whatever Python env has requirements-py311.txt
# installed (see RUN_ON_DIAS_HPC.md for the Apptainer-based setup).
set -euo pipefail

TEMP=${TEMP:-0.9}
MODEL_PATH=${MODEL_PATH:-"XueZhang-bjtu/1.5B-cold-start-SFT"}
MODEL_NAME=${MODEL_NAME:-"1.5B-cold-start-SFT"}
LANGS=${LANGS:-"en es fr pt"}
LEVELS=${LEVELS:-"low medium high top"}

# GPU memory/batch settings, sized for a 20GB card. Lower these (e.g. GPU_MEM_UTIL=0.35,
# MAX_MODEL_LEN=2048, MAX_TOKENS=512, MAX_NUM_SEQS=1, MAX_NUM_BATCHED_TOKENS=512, and add
# EXTRA_GEN_ARGS="--enforce_eager --disable_prefix_caching") if generation OOMs.
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.85}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-16384}
MAX_TOKENS=${MAX_TOKENS:-14336}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-64}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-32768}
EXTRA_GEN_ARGS=${EXTRA_GEN_ARGS:-}

# Fixed at 4: cal-polymath-acc.py's per-level breakdown hardcodes range(4) sample runs,
# and the scoring loop below iterates --cnt 0..3 to match.

export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}
export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}
export VLLM_DISABLE_FLASHINFER_PREFILL=${VLLM_DISABLE_FLASHINFER_PREFILL:-1}

echo "=== Generation ==="
for LVL in $LEVELS; do
  for L in $LANGS; do
    python3 eval_tools/PolyMath/polymath_res_gen.py \
      --lang "$L" --level "$LVL" --temp "$TEMP" \
      --model_path "$MODEL_PATH" --model_name "$MODEL_NAME" \
      --save_path "logs-eval/PolyMath-temp_${TEMP}" \
      --gpu_memory_utilization "$GPU_MEM_UTIL" --max_model_len "$MAX_MODEL_LEN" --max_tokens "$MAX_TOKENS" \
      --num_samples 4 --top_p 0.95 --max_num_seqs "$MAX_NUM_SEQS" --max_num_batched_tokens "$MAX_NUM_BATCHED_TOKENS" \
      $EXTRA_GEN_ARGS
  done
done

echo "=== Scoring ==="
rm -f "logs-eval/PolyMath-temp_${TEMP}/${MODEL_NAME}/score-eval.jsonl"
for L in $LANGS; do
  for LVL in $LEVELS; do
    for CNT in 0 1 2 3; do
      python3 eval_tools/PolyMath/eval/run_eval-fast.py \
        --model "$MODEL_NAME" --language "$L" --level "$LVL" --cnt "$CNT" \
        --lang_detector fasttext --fasttext_model_path eval_tools/langid/lid.176.ftz --fasttext_min_prob 0.2
    done
  done
done

echo "=== Per-level report ==="
python3 eval_tools/PolyMath/cal-polymath-acc.py --model_name "$MODEL_NAME" --langs $LANGS