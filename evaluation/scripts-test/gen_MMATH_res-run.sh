ROOT_PATH=eval_tools/MMATH
set -euo pipefail

temp=$1
MODEL_PATH=$2
MODEL_NAME=$3
PYTHON_BIN=${PYTHON_BIN:-python3}

if [ -z "$temp" ] || [ -z "$MODEL_PATH" ] || [ -z "$MODEL_NAME" ]; then
    echo "Usage: bash scripts-test/gen_MMATH_res-run.sh <temp> <model_path> <model_name>"
    exit 1
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "Error: Python executable '$PYTHON_BIN' not found."
    exit 1
fi

export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}
export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}
export VLLM_DISABLE_FLASHINFER_PREFILL=${VLLM_DISABLE_FLASHINFER_PREFILL:-1}

if [ -d "$MODEL_PATH/actor" ]; then
    "$PYTHON_BIN" verl/scripts/model_merger.py merge \
        --backend fsdp \
        --local_dir $MODEL_PATH/actor \
        --target_dir $MODEL_PATH
else
    echo "Skip model merge: $MODEL_PATH/actor does not exist."
fi


CUR_LOG=logs-eval/MMATH-temp_$temp/$MODEL_NAME/logs


if [ ! -d "$CUR_LOG" ]; then
    mkdir -p $CUR_LOG
fi
chmod 777 -R $CUR_LOG

detect_gpu_ids() {
    if [ -n "${GPU_IDS:-}" ]; then
        echo "$GPU_IDS"
        return
    fi

    if command -v nvidia-smi >/dev/null 2>&1; then
        local ids
        ids=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | paste -sd, -)
        if [ -n "$ids" ]; then
            echo "$ids"
            return
        fi
    fi

    echo "0"
}

GPU_IDS_STR=$(detect_gpu_ids)
IFS=',' read -r -a GPU_LIST <<< "$GPU_IDS_STR"
if [ ${#GPU_LIST[@]} -eq 0 ]; then
    GPU_LIST=(0)
fi

if [ -z "${SERIAL:-}" ]; then
    if [ ${#GPU_LIST[@]} -eq 1 ]; then
        SERIAL=1
    else
        SERIAL=0
    fi
fi

MMATH_ARGS=()
if [ -n "${MMATH_EXTRA_ARGS:-}" ]; then
    read -r -a MMATH_ARGS <<< "$MMATH_EXTRA_ARGS"
fi

LANG_GROUPS=("en ko" "zh ja" "ar" "fr" "es" "pt" "th" "vi")
LOG_NAMES=("en-ko" "zh-ja" "ar" "fr" "es" "pt" "th" "vi")

if [ -n "${MMATH_LANGS:-}" ]; then
    MMATH_LANGS_CLEAN=${MMATH_LANGS//,/ }
    read -r -a MMATH_LANG_ARRAY <<< "$MMATH_LANGS_CLEAN"
    LANG_GROUPS=()
    LOG_NAMES=()
    for lang in "${MMATH_LANG_ARRAY[@]}"; do
        LANG_GROUPS+=("$lang")
        LOG_NAMES+=("$lang")
    done
fi
echo "MMATH language groups: ${LANG_GROUPS[*]}"

run_job() {
    local gpu_id=$1
    local lang_group=$2
    local log_name=$3
    local log_file=$CUR_LOG/${log_name}.log
    local t0
    local t1
    local elapsed
    read -r -a lang_args <<< "$lang_group"

    t0=$(date +%s)
    if [ "${MMATH_STREAM_LOG:-0}" = "1" ]; then
        if ! CUDA_VISIBLE_DEVICES=$gpu_id "$PYTHON_BIN" $ROOT_PATH/mmath_eval.py \
            --lang "${lang_args[@]}" \
            --temp $temp \
            --model_path $MODEL_PATH \
            --model_name $MODEL_NAME \
            "${MMATH_ARGS[@]}" \
            2>&1 | tee "$log_file"; then
            echo "MMATH job failed for language group '$lang_group' on GPU $gpu_id."
            echo "Log: $log_file"
            tail -n 80 "$log_file" || true
            return 1
        fi
    elif ! CUDA_VISIBLE_DEVICES=$gpu_id "$PYTHON_BIN" $ROOT_PATH/mmath_eval.py \
        --lang "${lang_args[@]}" \
        --temp $temp \
        --model_path $MODEL_PATH \
        --model_name $MODEL_NAME \
        "${MMATH_ARGS[@]}" \
        > "$log_file" 2>&1; then
        echo "MMATH job failed for language group '$lang_group' on GPU $gpu_id."
        echo "Log: $log_file"
        tail -n 80 "$log_file" || true
        return 1
    fi
    t1=$(date +%s)
    elapsed=$((t1 - t0))
    echo "MMATH done: group='$lang_group' gpu=$gpu_id elapsed=${elapsed}s log=$log_file"
    if grep -q "\[TIMING\]" "$log_file"; then
        grep "\[TIMING\]" "$log_file" | sed 's/^/  /'
    fi
}

if [ "$SERIAL" = "1" ]; then
    echo "Running MMATH sequentially on GPU ${GPU_LIST[0]}"
    for i in "${!LANG_GROUPS[@]}"; do
        run_job "${GPU_LIST[0]}" "${LANG_GROUPS[$i]}" "${LOG_NAMES[$i]}"
    done
else
    echo "Running MMATH with ${#GPU_LIST[@]} GPUs: ${GPU_LIST[*]}"
    for ((start=0; start<${#LANG_GROUPS[@]}; start+=${#GPU_LIST[@]})); do
        for ((slot=0; slot<${#GPU_LIST[@]}; slot++)); do
            idx=$((start + slot))
            if [ $idx -ge ${#LANG_GROUPS[@]} ]; then
                break
            fi
            gpu_id=${GPU_LIST[$slot]}
            lang_group=${LANG_GROUPS[$idx]}
            log_name=${LOG_NAMES[$idx]}
            read -r -a lang_args <<< "$lang_group"
            CUDA_VISIBLE_DEVICES=$gpu_id nohup "$PYTHON_BIN" $ROOT_PATH/mmath_eval.py \
                --lang "${lang_args[@]}" \
                --temp $temp \
                --model_path $MODEL_PATH \
                --model_name $MODEL_NAME \
                "${MMATH_ARGS[@]}" \
                > $CUR_LOG/${log_name}.log 2>&1 &
        done
        wait
    done
fi

sleep 5s
