temp=$1

MODEL_PATH=$2
MODEL_NAME=$3
# from 320 ~ 435 (24 checkpoints)
# 320 325 330 335 340 345 350 355
# 360 365 370 375 380 385 390 395
# 400 405 410 415 420 425 430 435

if [ -z "$temp" ] || [ -z "$MODEL_PATH" ] || [ -z "$MODEL_NAME" ]; then
    echo "Usage: bash scripts-test/eval_MMATH_multiple_steps.sh <temp> <model_path> <model_name>"
    exit 1
fi

if [ -n "${STEP_LIST:-}" ]; then
    IFS=',' read -r -a all_steps <<< "$STEP_LIST"
else
    all_steps=(320 325 330 335 340 345 350 355 360 365 370 375 380 385 390 395 400 405 410 415 420 425 430 435)
fi

steps=()
if [ -z "${RANK:-}" ]; then
    # Single-machine fallback: run all checkpoints sequentially.
    steps=("${all_steps[@]}")
else
    # Multi-node mode: distribute checkpoints by rank (0-7).
    for i in "${!all_steps[@]}"; do
        if [ $((i % 8)) -eq "$RANK" ]; then
            steps+=("${all_steps[$i]}")
        fi
    done
fi

for step in "${steps[@]}"; do
    bash scripts-test/gen_MMATH_res-run.sh $temp ${MODEL_PATH}/global_step_${step} ${MODEL_NAME}/step${step}-temp_$temp
    wait
done

