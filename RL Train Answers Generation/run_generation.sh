#!/bin/bash

# ==========================================
# 1. SET DEFAULT VALUES
# These are used if you don't provide a flag
# ==========================================
GPU_ID="1"
BASE_DIR=""
FAKE_HOME=""
INPUT_FILE="input_questions.jsonl"
OUTPUT_FILE="dpo_candidates_output.jsonl"

# ==========================================
# 2. PARSE COMMAND LINE ARGUMENTS
# ==========================================
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --gpu) GPU_ID="$2"; shift ;;
        --home) FAKE_HOME="$2"; shift ;;
        --base_dir) BASE_DIR="$2"; shift ;;
        --input) INPUT_FILE="$2"; shift ;;
        --output) OUTPUT_FILE="$2"; shift ;;
        *) echo "❌ Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

echo "========================================"
echo "🚀 LAUNCHING GENERATION JOB"
echo "========================================"
echo "GPU ID:      $GPU_ID"
echo "FAKE HOME:   $FAKE_HOME"
echo "INPUT FILE:  $INPUT_FILE"
echo "OUTPUT FILE: $OUTPUT_FILE"
echo "========================================"

# ==========================================
# 3. SET ENVIRONMENT VARIABLES
# ==========================================
# Target the specific GPU
export CUDA_VISIBLE_DEVICES="$GPU_ID"

export VLLM_USE_V1=0

# Disable vLLM telemetry
export VLLM_NO_USAGE_STATS=1

# Protect your quota from hidden cache files
export HOME="$FAKE_HOME"
mkdir -p "$HOME"

# Reroute Hugging Face weights explicitly
export HF_HOME="$BASE_DIR/hf_cache"

# ==========================================
# 4. ACTIVATE ENVIRONMENT & RUN
# ==========================================
source "$BASE_DIR/generation_venv/bin/activate"

# Run the python script with the parsed arguments
python3 generate_dpo_data.py \
    --base_dir "$BASE_DIR" \
    --input_file "$INPUT_FILE" \
    --output_file "$OUTPUT_FILE" 
