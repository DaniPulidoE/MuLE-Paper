model_name=$1
model_list=($model_name)
MAX_PARALLEL_JOBS=${MAX_PARALLEL_JOBS:-4}
PYTHON_BIN=${PYTHON_BIN:-python3}

if [ -z "$model_name" ]; then
    echo "Usage: bash eval_tools/PolyMath/eval/run_eval.sh <model_name>"
    exit 1
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "Error: Python executable '$PYTHON_BIN' not found."
    exit 1
fi

# language_list=(en zh ar bn de es fr id it ja ko ms pt ru sw te th vi)

language_list=(ko ja pt th en zh ar es fr vi)
level_list=(low medium high top)
cnt_array=(0 1 2 3)

for cnt in ${cnt_array[@]};
do  
    export PYTHONPATH=eval_tools/PolyMath/eval
    for i in ${model_list[*]}; do
        mkdir -p logs-eval/PolyMath-temp_0.9/$i
        for j in ${language_list[*]}; do
            for k in ${level_list[*]}; do
                "$PYTHON_BIN" eval_tools/PolyMath/eval/run_eval-fast.py --model $i --language $j --level $k --cnt $cnt >> logs-eval/PolyMath-temp_0.9/$i/eval.log 2>&1 &
                while [ "$(jobs -pr | wc -l | tr -d ' ')" -ge "$MAX_PARALLEL_JOBS" ]; do
                    sleep 2s
                done
            done
        done
    done
done

echo "waiting..."
wait 
echo "Done!"
