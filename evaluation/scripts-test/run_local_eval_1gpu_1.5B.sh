#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    echo "Usage: bash scripts-test/run_local_eval_1gpu_1.5B.sh [temp] [model_path] [model_name]"
    echo "Example: bash scripts-test/run_local_eval_1gpu_1.5B.sh 0.9 XueZhang-bjtu/1.5B-cold-start-SFT 1.5B-cold-start-SFT"
    exit 0
fi

TEMP=${1:-0.9}
MODEL_PATH=${2:-XueZhang-bjtu/1.5B-cold-start-SFT}
MODEL_NAME=${3:-1.5B-cold-start-SFT}
PYTHON_BIN=${PYTHON_BIN:-python3}

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "Error: Python executable '$PYTHON_BIN' not found."
    echo "Set PYTHON_BIN explicitly, for example:"
    echo "PYTHON_BIN=python3.11 bash scripts-test/run_local_eval_1gpu_1.5B.sh"
    exit 1
fi

# Force single-GPU sequential execution in patched scripts.
export GPU_IDS=${GPU_IDS:-0}
export SERIAL=${SERIAL:-1}
export MAX_PARALLEL_JOBS=${MAX_PARALLEL_JOBS:-1}
export CLEAN_OUTPUT=${CLEAN_OUTPUT:-1}
export ONLY_MMATH=${ONLY_MMATH:-0}
export MMATH_EXTRA_ARGS=${MMATH_EXTRA_ARGS:---gpu_memory_utilization 0.35 --max_model_len 2048 --max_tokens 512 --max_num_seqs 1 --max_num_batched_tokens 512 --enforce_eager --disable_prefix_caching}
export POLYMATH_EXTRA_ARGS=${POLYMATH_EXTRA_ARGS:---gpu_memory_utilization 0.35 --max_model_len 2048 --max_tokens 512 --max_num_seqs 1 --max_num_batched_tokens 512 --enforce_eager --disable_prefix_caching}
# Avoid FlashInfer JIT paths on shared clusters where nvcc/CUDA toolkit is unavailable.
export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}
export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}
export VLLM_DISABLE_FLASHINFER_PREFILL=${VLLM_DISABLE_FLASHINFER_PREFILL:-1}
export PYTHON_BIN

echo "Python executable: $(command -v "$PYTHON_BIN")"
echo "Python version: $("$PYTHON_BIN" -V)"
echo "PYTHON_BIN=$PYTHON_BIN"

PY_VER=$("$PYTHON_BIN" - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)
"$PYTHON_BIN" - <<'PY'
import sys
if sys.version_info < (3, 10):
    raise SystemExit(
        "Error: Python >= 3.10 is required for this workflow "
        f"(current: {sys.version_info.major}.{sys.version_info.minor})."
    )
PY
if [[ "$PY_VER" == "3.13" ]]; then
    echo "Warning: Python 3.13 can be unstable with some ML packages in this repo."
    echo "Recommended: use Python 3.10/3.11 conda env for reproducibility."
fi

check_module() {
    local module_name=$1
    "$PYTHON_BIN" - <<PY
import importlib.util
import sys
sys.exit(0 if importlib.util.find_spec("${module_name}") else 1)
PY
}

MISSING_MODULES=()
for module in vllm transformers datasets math_verify langdetect jsonlines numpy tqdm; do
    if ! check_module "$module"; then
        MISSING_MODULES+=("$module")
    fi
done

if [ ${#MISSING_MODULES[@]} -ne 0 ]; then
    echo "Missing Python modules: ${MISSING_MODULES[*]}"
    echo "Install them in your active Python 3.11 environment, for example:"
    if [ -f requirements-py311.txt ]; then
        echo "$PYTHON_BIN -m pip install -r requirements-py311.txt"
    else
        echo "$PYTHON_BIN -m pip install vllm transformers datasets math-verify langdetect jsonlines numpy tqdm"
    fi
    exit 1
fi

assert_mmath_outputs() {
    local base_dir=$1
    local langs_raw=${MMATH_LANGS:-"en zh ar es fr ja ko pt th vi"}
    local langs_clean=${langs_raw//,/ }
    read -r -a langs <<< "$langs_clean"
    local missing=()
    for lang in "${langs[@]}"; do
        if [ ! -f "$base_dir/$lang.json" ]; then
            missing+=("$lang.json")
        fi
    done

    if [ ${#missing[@]} -ne 0 ]; then
        echo "MMATH generation did not produce expected files: ${missing[*]}"
        echo "Check logs under: $base_dir/logs"
        exit 1
    fi
}

assert_polymath_outputs() {
    local base_dir=$1
    local levels=(low medium high top)
    local langs=(ko ja pt th en zh ar es fr vi)
    local missing=()

    for level in "${levels[@]}"; do
        for lang in "${langs[@]}"; do
            if [ ! -f "$base_dir/$level/$lang.json" ]; then
                missing+=("$level/$lang.json")
            fi
        done
    done

    if [ ${#missing[@]} -ne 0 ]; then
        echo "PolyMath generation did not produce expected files."
        echo "Missing sample(s): ${missing[0]}"
        echo "Check logs under: $base_dir/logs"
        exit 1
    fi
}

echo "Running local 1-GPU eval"
echo "TEMP=$TEMP"
echo "MODEL_PATH=$MODEL_PATH"
echo "MODEL_NAME=$MODEL_NAME"
echo "GPU_IDS=$GPU_IDS"
echo "CLEAN_OUTPUT=$CLEAN_OUTPUT"
echo "ONLY_MMATH=$ONLY_MMATH"
echo "MMATH_LANGS=${MMATH_LANGS:-all(default)}"
echo "MMATH_EXTRA_ARGS=$MMATH_EXTRA_ARGS"
echo "POLYMATH_EXTRA_ARGS=$POLYMATH_EXTRA_ARGS"
echo "VLLM_ATTENTION_BACKEND=$VLLM_ATTENTION_BACKEND"
echo "VLLM_USE_FLASHINFER_SAMPLER=$VLLM_USE_FLASHINFER_SAMPLER"
echo "VLLM_DISABLE_FLASHINFER_PREFILL=$VLLM_DISABLE_FLASHINFER_PREFILL"

if [ "$CLEAN_OUTPUT" = "1" ]; then
    rm -rf "logs-eval/MMATH-temp_${TEMP}/${MODEL_NAME}"
    rm -rf "logs-eval/PolyMath-temp_${TEMP}/${MODEL_NAME}"
fi

echo "[1/5] MMATH generation"
bash scripts-test/gen_MMATH_res.sh "$TEMP" "$MODEL_PATH" "$MODEL_NAME"
assert_mmath_outputs "logs-eval/MMATH-temp_${TEMP}/${MODEL_NAME}"

echo "[2/5] MMATH scoring"
MMATH_SCORE_ARGS=()
if [ -n "${MMATH_LANGS:-}" ]; then
    MMATH_LANGS_CLEAN=${MMATH_LANGS//,/ }
    read -r -a MMATH_LANG_ARRAY <<< "$MMATH_LANGS_CLEAN"
    MMATH_SCORE_ARGS+=(--langs "${MMATH_LANG_ARRAY[@]}")
fi
"$PYTHON_BIN" eval_tools/MMATH/cal-MMATH-acc.py --res_path "logs-eval/MMATH-temp_${TEMP}/${MODEL_NAME}" "${MMATH_SCORE_ARGS[@]}"

if [ "$ONLY_MMATH" = "1" ]; then
    echo "ONLY_MMATH=1, skipping PolyMath steps."
    echo "Done."
    exit 0
fi

echo "[3/5] PolyMath generation"
bash scripts-test/gen_PolyMath_res.sh "$TEMP" "$MODEL_PATH" "$MODEL_NAME"
assert_polymath_outputs "logs-eval/PolyMath-temp_${TEMP}/${MODEL_NAME}"

echo "[4/5] PolyMath per-split scoring"
bash eval_tools/PolyMath/eval/run_eval.sh "$MODEL_NAME"

echo "[5/5] PolyMath aggregate scoring"
"$PYTHON_BIN" eval_tools/PolyMath/cal-polymath-acc.py --model_name "$MODEL_NAME"

echo "Done."
