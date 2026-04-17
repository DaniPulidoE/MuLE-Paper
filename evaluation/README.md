### Quick Evaluation (Generation + Scoring)

This section provides a minimal end-to-end flow for MMATH and PolyMath on `en/es/fr/pt`.

```bash
# 1) Environment
python3 -m venv generation_venv
source generation_venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-py311.txt
python -m pip install fasttext-wheel peft

# 2) FastText language ID model (required for LC with fasttext backend)
mkdir -p eval_tools/langid
curl -L https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz -o eval_tools/langid/lid.176.ftz
```

If your checkpoint is a LoRA adapter, merge it first (replace adp name as needed):

```bash
python3 -c "from transformers import AutoModelForCausalLM,AutoTokenizer; from peft import PeftModel; import torch; base='XueZhang-bjtu/1.5B-cold-start-SFT'; adp='azure_work/trial_no_precomp_v18'; out='azure_work/merged_trial_no_precomp_v18'; tok=AutoTokenizer.from_pretrained(base,trust_remote_code=True); m=AutoModelForCausalLM.from_pretrained(base,torch_dtype=torch.float16,device_map='auto',trust_remote_code=True); m=PeftModel.from_pretrained(m,adp).merge_and_unload(); m.save_pretrained(out); tok.save_pretrained(out); print('saved',out)"
```

Recommended variable setup:

```bash
MODEL_PATH="XueZhang-bjtu/1.5B-cold-start-SFT"   # or merged adapter path
MODEL_NAME="sft-eval-en-es-fr-pt"                # output folder name
TEMP=0.9
```

MMATH generation (full benchmark, 4 languages, adjust to your hardware):

```bash
python3 eval_tools/MMATH/mmath_eval.py \
  --model_path "$MODEL_PATH" \
  --model_name "$MODEL_NAME" \
  --temp $TEMP \
  --lang en es fr pt \
  --gpu_memory_utilization 0.95 \
  --max_model_len 16384 \
  --max_tokens 14336 \
  --num_samples 4 \
  --top_p 0.95 \
  --max_num_seqs 128 \
  --max_num_batched_tokens 65536 \
  --log_lang_timing
```

MMATH generation on validation subset (instead of full benchmark):

```bash
python3 eval_tools/MMATH/mmath_eval.py \
  --model_path "$MODEL_PATH" \
  --model_name "$MODEL_NAME" \
  --temp $TEMP \
  --lang en es fr pt \
  --input_jsonl val_set_mMath.jsonl \
  --gpu_memory_utilization 0.95 \
  --max_model_len 16384 \
  --max_tokens 14336 \
  --num_samples 4 \
  --top_p 0.95 \
  --max_num_seqs 128 \
  --max_num_batched_tokens 65536 \
  --log_lang_timing
```

MMATH scoring:

```bash
python3 eval_tools/MMATH/cal-MMATH-acc.py \
  --res_path logs-eval/MMATH-temp_${TEMP}/${MODEL_NAME} \
  --num_samples 4 \
  --tokenizer_path "$MODEL_PATH" \
  --langs en es fr pt \
  --lang_detector fasttext \
  --fasttext_model_path eval_tools/langid/lid.176.ftz \
  --fasttext_min_prob 0.2 \
  --report_variance
```

PolyMath generation (full benchmark, 4 languages x 4 levels):

```bash
for LVL in low medium high top; do
  for L in en es fr pt; do
    python3 eval_tools/PolyMath/polymath_res_gen.py \
      --lang "$L" \
      --level "$LVL" \
      --temp $TEMP \
      --model_path "$MODEL_PATH" \
      --model_name "$MODEL_NAME" \
      --save_path logs-eval/PolyMath-temp_${TEMP} \
      --gpu_memory_utilization 0.95 \
      --max_model_len 16384 \
      --max_tokens 14336 \
      --num_samples 4 \
      --top_p 0.95 \
      --max_num_seqs 128 \
      --max_num_batched_tokens 65536
  done
done
```

PolyMath generation on validation subset (adjust settings according to your hardware):

```bash
for LVL in low medium high top; do
  for L in en es fr pt; do
    python3 eval_tools/PolyMath/polymath_res_gen.py \
      --lang "$L" \
      --level "$LVL" \
      --temp $TEMP \
      --model_path "$MODEL_PATH" \
      --model_name "$MODEL_NAME" \
      --save_path logs-eval/PolyMath-temp_${TEMP} \
      --input_jsonl val_set_polyMath.jsonl \
      --gpu_memory_utilization 0.95 \
      --max_model_len 16384 \
      --max_tokens 14336 \
      --num_samples 4 \
      --top_p 0.95 \
      --max_num_seqs 128 \
      --max_num_batched_tokens 65536
  done
done
```

PolyMath scoring and aggregate table:

```bash
rm -f logs-eval/PolyMath-temp_${TEMP}/${MODEL_NAME}/score-eval.jsonl
for L in en es fr pt; do
  for LVL in low medium high top; do
    for CNT in 0 1 2 3; do
      python3 eval_tools/PolyMath/eval/run_eval-fast.py \
        --model "$MODEL_NAME" \
        --language "$L" \
        --level "$LVL" \
        --cnt "$CNT" \
        --lang_detector fasttext \
        --fasttext_model_path eval_tools/langid/lid.176.ftz \
        --fasttext_min_prob 0.2
    done
  done
done
python3 eval_tools/PolyMath/eval/run_eval-fast.py --model "$MODEL_NAME" --report_table --report_variance
```

Result locations:

- MMATH generations: `logs-eval/MMATH-temp_<temp>/<model_name>/<lang>.json`
- PolyMath generations: `logs-eval/PolyMath-temp_<temp>/<model_name>/<level>/<lang>.json`
- PolyMath score rows: `logs-eval/PolyMath-temp_<temp>/<model_name>/score-eval.jsonl`


### Single-GPU Local Evaluation (1.5B Example)

The training scripts are designed for multi-node/multi-GPU clusters. For local one-GPU usage, we recommend evaluating released checkpoints.

The scripts below auto-detect available GPUs and run sequentially when only one GPU is found.
If your default shell is `csh/tcsh`, switch to `bash` before running these commands.

```
# Recommended: Python 3.11.x environment (but other environments can work)
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-py311.txt

# Optional: force scripts to use your Python 3.11 binary
export PYTHON_BIN=python3.11

# One-command local eval (defaults to 1.5B cold-start SFT)
bash scripts-test/run_local_eval_1gpu_1.5B.sh

# If GPU memory is tight, keep/override conservative defaults:
# export MMATH_EXTRA_ARGS="--gpu_memory_utilization 0.35 --max_model_len 2048 --max_tokens 512 --max_num_seqs 1 --max_num_batched_tokens 512 --enforce_eager --disable_prefix_caching"
# export POLYMATH_EXTRA_ARGS="--gpu_memory_utilization 0.35 --max_model_len 2048 --max_tokens 512 --max_num_seqs 1 --max_num_batched_tokens 512 --enforce_eager --disable_prefix_caching"
# On shared clusters without CUDA toolkit/nvcc, force FlashAttention backend:
# export VLLM_ATTENTION_BACKEND=FLASH_ATTN
# export VLLM_USE_FLASHINFER_SAMPLER=0
# export VLLM_DISABLE_FLASHINFER_PREFILL=1

# MMATH generation + scoring
bash scripts-test/gen_MMATH_res.sh 0.9 XueZhang-bjtu/1.5B-cold-start-SFT 1.5B-cold-start-SFT
python eval_tools/MMATH/cal-MMATH-acc.py --res_path logs-eval/MMATH-temp_0.9/1.5B-cold-start-SFT

# PolyMath generation + scoring
bash scripts-test/gen_PolyMath_res.sh 0.9 XueZhang-bjtu/1.5B-cold-start-SFT 1.5B-cold-start-SFT
bash eval_tools/PolyMath/eval/run_eval.sh 1.5B-cold-start-SFT
python eval_tools/PolyMath/cal-polymath-acc.py --model_name 1.5B-cold-start-SFT
```

If you want to force specific GPUs, set `GPU_IDS` before running scripts:

```
export GPU_IDS=0
```


### Acknowledgement

We gratefully acknowledge [MMATH](https://github.com/RUCAIBox/MMATH) and [PolyMath](https://github.com/qwenlm/polymath) for their significant contributions to evaluating the multilingual reasoning capabilities of models.

### Citation
If you find this work useful, please consider citing our paper:

```
@misc{zhang2025thinknativelyunlockingmultilingual,
      title={Think Natively: Unlocking Multilingual Reasoning with Consistency-Enhanced Reinforcement Learning}, 
      author={Xue Zhang and Yunlong Liang and Fandong Meng and Songming Zhang and Kaiyu Huang and Yufeng Chen and Jinan Xu and Jie Zhou},
      year={2025},
      eprint={2510.07300},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2510.07300}, 
}
```
