ROOT_PATH=eval_tools/PolyMath
set -euo pipefail


temp=$1
MODEL_PATH=$2
MODEL_NAME=$3
PYTHON_BIN=${PYTHON_BIN:-python3}
CUR_LOG=logs-eval/PolyMath-temp_$temp/$MODEL_NAME/logs

if [ -z "$temp" ] || [ -z "$MODEL_PATH" ] || [ -z "$MODEL_NAME" ]; then
    echo "Usage: bash scripts-test/gen_PolyMath_res.sh <temp> <model_path> <model_name>"
    exit 1
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "Error: Python executable '$PYTHON_BIN' not found."
    exit 1
fi

export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}
export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}
export VLLM_DISABLE_FLASHINFER_PREFILL=${VLLM_DISABLE_FLASHINFER_PREFILL:-1}

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

POLYMATH_ARGS=()
if [ -n "${POLYMATH_EXTRA_ARGS:-}" ]; then
    read -r -a POLYMATH_ARGS <<< "$POLYMATH_EXTRA_ARGS"
fi

run_job() {
    local gpu_id=$1
    local lang=$2
    local level=$3
    local log_file=$CUR_LOG/${lang}_${level}.log
    if ! CUDA_VISIBLE_DEVICES=$gpu_id "$PYTHON_BIN" $ROOT_PATH/polymath_res_gen.py \
        --lang $lang \
        --level $level \
        --temp $temp \
        --model_path $MODEL_PATH \
        --model_name $MODEL_NAME \
        "${POLYMATH_ARGS[@]}" \
        > "$log_file" 2>&1; then
        echo "PolyMath job failed for lang=$lang level=$level on GPU $gpu_id."
        echo "Log: $log_file"
        tail -n 80 "$log_file" || true
        return 1
    fi
}

level_array=(low medium high top)
lang_array=(es fr ar ja ko pt th vi)

if [ "$SERIAL" = "1" ]; then
    echo "Running PolyMath sequentially on GPU ${GPU_LIST[0]}"
    for level in ${level_array[@]}; do
        for lang in ${lang_array[@]}; do
            run_job "${GPU_LIST[0]}" "$lang" "$level"
        done
    done
    for level in ${level_array[@]}; do
        run_job "${GPU_LIST[0]}" "en" "$level"
        run_job "${GPU_LIST[0]}" "zh" "$level"
    done
else
    echo "Running PolyMath with ${#GPU_LIST[@]} GPUs: ${GPU_LIST[*]}"
    for level in ${level_array[@]}; do
        for ((start=0; start<${#lang_array[@]}; start+=${#GPU_LIST[@]})); do
            for ((slot=0; slot<${#GPU_LIST[@]}; slot++)); do
                idx=$((start + slot))
                if [ $idx -ge ${#lang_array[@]} ]; then
                    break
                fi
                gpu_id=${GPU_LIST[$slot]}
                lang=${lang_array[$idx]}
                CUDA_VISIBLE_DEVICES=$gpu_id nohup "$PYTHON_BIN" $ROOT_PATH/polymath_res_gen.py \
                    --lang $lang \
                    --level $level \
                    --temp $temp \
                    --model_path $MODEL_PATH \
                    --model_name $MODEL_NAME \
                    "${POLYMATH_ARGS[@]}" \
                    > $CUR_LOG/${lang}_${level}.log 2>&1 &
            done
            wait
        done
    done

    enzh_tasks=("en low" "en medium" "en high" "en top" "zh low" "zh medium" "zh high" "zh top")
    for ((start=0; start<${#enzh_tasks[@]}; start+=${#GPU_LIST[@]})); do
        for ((slot=0; slot<${#GPU_LIST[@]}; slot++)); do
            idx=$((start + slot))
            if [ $idx -ge ${#enzh_tasks[@]} ]; then
                break
            fi
            gpu_id=${GPU_LIST[$slot]}
            lang_level=${enzh_tasks[$idx]}
            read -r lang level <<< "$lang_level"
            CUDA_VISIBLE_DEVICES=$gpu_id nohup "$PYTHON_BIN" $ROOT_PATH/polymath_res_gen.py \
                --lang $lang \
                --level $level \
                --temp $temp \
                --model_path $MODEL_PATH \
                --model_name $MODEL_NAME \
                "${POLYMATH_ARGS[@]}" \
                > $CUR_LOG/${lang}_${level}.log 2>&1 &
        done
        wait
    done
fi


sleep 5s
