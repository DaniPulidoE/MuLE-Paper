import os

# Set writable cache directories BEFORE any imports that use datasets/transformers
os.environ["TRANSFORMERS_CACHE"] = "/tmp/hf_cache"
os.environ["HF_DATASETS_CACHE"] = "/tmp/hf_datasets_cache"
os.makedirs("/tmp/hf_cache", exist_ok=True)
os.makedirs("/tmp/hf_datasets_cache", exist_ok=True)

os.environ["HF_HOME"] = "/tmp/hf_home"
os.environ["WANDB_CONFIG_DIR"] = "/tmp/wandb_config"
os.makedirs("/tmp/hf_home", exist_ok=True)
os.makedirs("/tmp/wandb_config", exist_ok=True)

os.system("pip install datasets transformers unsloth trl bitsandbytes wandb")

import json
import argparse
import logging
import warnings
import gc

warnings.filterwarnings("ignore")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import wandb
import datasets
from datasets import Dataset

datasets.config.HF_DATASETS_CACHE = "/tmp/hf_datasets_cache"
datasets.config.IN_MEMORY_MAX_SIZE = 0
from unsloth import FastLanguageModel, PatchDPOTrainer
from trl import DPOTrainer, DPOConfig
from transformers import TrainerCallback
import time

MODEL_NAME = "XueZhang-bjtu/1.5B-cold-start-SFT"

CONFIG = {
    "dataset_file":     "./dpo_dataset_fixed.jsonl",
    "stats_file":       "./stats.json",
    "output_dir":       "./dpo_output",
    "wandb_project":    "mule-dpo",
    "wandb_run_name":   "dpo-run1",
    "beta":             0.1,
    "learning_rate":    1e-6,
    "num_epochs":       1,
    "batch_size":       1,
    "grad_accum_steps": 64,
    "max_length":       16384,
    "max_pairs":        None,
    "cpu_test":         False,
}


def load_dpo_dataset(dataset_file, cpu_test=False, max_pairs=None):
    pairs = []
    with open(dataset_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                if "prompt" not in record or "chosen" not in record or "rejected" not in record:
                    logging.warning(f"Line {line_num}: missing field, skipping.")
                    continue
                pairs.append(record)
            except json.JSONDecodeError as e:
                logging.warning(f"Line {line_num}: JSON error ({e}), skipping.")
                continue

    total_loaded = len(pairs)

    if cpu_test:
        pairs.sort(key=lambda p: len(p["prompt"]) + len(p["chosen"]) + len(p["rejected"]))
        pairs = pairs[:1]
        logging.info("CPU TEST MODE: using only the 1 shortest pair")
    elif max_pairs is not None and max_pairs < total_loaded:
        pairs.sort(key=lambda p: len(p["prompt"]) + len(p["chosen"]) + len(p["rejected"]))
        pairs = pairs[:max_pairs]
        logging.info(f"Using {max_pairs} shortest pairs out of {total_loaded} total")

    return pairs


def load_stats(stats_file):
    if not os.path.exists(stats_file):
        logging.warning(f"Stats file not found: {stats_file}")
        return {}
    with open(stats_file, "r") as f:
        return json.load(f)


def get_latest_checkpoint(output_dir):
    if output_dir is None:
        return None
    if not os.path.isdir(output_dir):
        return None
    checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
    if not checkpoints:
        return None
    latest = sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))[-1]
    return os.path.join(output_dir, latest)


class GPUMemoryWatchdog(TrainerCallback):
    def __init__(self, check_interval=1):
        self.check_interval = check_interval

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.check_interval != 0:
            return control
        if not torch.cuda.is_available():
            return control
        free, total = torch.cuda.mem_get_info()
        used = total - free
        utilization = used / total
        print(f"\nGPU memory usage: {utilization*100:.1f}% used")
        return control


def main(args):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    if args.cpu_test:
        logger.info("=" * 55)
        logger.info("  CPU TEST MODE")
        logger.info("  1 pair, 1 epoch, max_length=256")
        logger.info("=" * 55)
    else:
        logger.info("=" * 55)
        logger.info("  FULL TRAINING MODE (no precompute)")
        logger.info(f"  max_pairs={args.max_pairs}, max_length={args.max_length}")
        logger.info("=" * 55)

    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config=vars(args),
    )
    logger.info(f"W&B run started: {args.wandb_run_name}")

    logger.info(f"Loading model: {MODEL_NAME}")
    PatchDPOTrainer()
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=args.max_length,
        load_in_4bit=True,
        attn_implementation="sdpa",
        dtype=torch.bfloat16,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth",
    )
    model = FastLanguageModel.for_training(model)
    model.config.use_cache = False

    # ── Load raw JSONL pairs (no precompute) ──────────────────────────────
    logger.info(f"Loading raw DPO pairs from: {args.dataset_file}")
    max_pairs = None if args.cpu_test else args.max_pairs
    pairs = load_dpo_dataset(args.dataset_file, cpu_test=args.cpu_test, max_pairs=max_pairs)
    if len(pairs) == 0:
        raise ValueError("No pairs loaded.")
    train_dataset = Dataset.from_list(pairs)
    ref_model = None
    precompute_ref_log_probs = True
    # ──────────────────────────────────────────────────────────────────────

    logger.info(f"Loaded {len(train_dataset)} preference pairs.")
    logger.info("--- Sanity check: first pair ---")
    first = train_dataset[0]
    logger.info(f"PROMPT   (first 150 chars): {first['prompt'][:150]}")
    logger.info(f"CHOSEN   (first 150 chars): {first['chosen'][:150]}")
    logger.info(f"REJECTED (first 150 chars): {first['rejected'][:150]}")

    stats = load_stats(args.stats_file)
    if stats:
        for k, v in stats.items():
            if not isinstance(v, list):
                wandb.run.summary[f"dataset/{k}"] = v
        if stats.get("total_questions", 0) > 0:
            wandb.run.summary["dataset/pair_yield_pct"] = (
                100 * stats.get("pairs_created", 0) / stats["total_questions"]
            )

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info(f"Train: {len(train_dataset)} pairs")

    wandb.run.summary["dataset/train_pairs"] = len(train_dataset)
    wandb.run.summary["dataset/eval_pairs"] = 0

    num_epochs = 1    if args.cpu_test else args.num_epochs
    max_length  = 256 if args.cpu_test else args.max_length
    grad_accum  = 1   if args.cpu_test else args.grad_accum_steps

    logger.info(f"Config: epochs={num_epochs}, max_length={max_length}, grad_accum={grad_accum}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        free_gb  = torch.cuda.mem_get_info()[0] / 1e9
        total_gb = torch.cuda.mem_get_info()[1] / 1e9
        logger.info(f"GPU memory at start: {free_gb:.1f}GB free / {total_gb:.1f}GB total")

    dpo_config = DPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=24,
        beta=args.beta,
        loss_type="sigmoid",
        max_length=max_length,
        truncation_mode="keep_end",
        precompute_ref_log_probs=precompute_ref_log_probs,
        logging_steps=1,
        report_to="wandb",
        run_name=args.wandb_run_name,
        eval_strategy="no",
        bf16=True,
        fp16=False,
        use_cpu=args.cpu_test,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        seed=24,
        disable_dropout=True,
        torch_empty_cache_steps=1,
        optim="adamw_8bit",
        dataloader_num_workers=1,
        dataloader_pin_memory=False,
        save_strategy="steps",
        save_steps=1,
        save_total_limit=2,
        max_grad_norm=2.0,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        callbacks=[GPUMemoryWatchdog(check_interval=1)]
    )

    checkpoint = get_latest_checkpoint(args.checkpoint_dir)
    if checkpoint:
        print(f"🔁 Resuming from {checkpoint}")
    else:
        print("🆕 Starting fresh training")

    logger.info("Starting DPO training...")
    logger.info("(precompute_ref_log_probs=True: ref model runs once then is freed)")
    train_result = trainer.train(resume_from_checkpoint=checkpoint)

    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    final_model_path = os.path.join(args.output_dir, f"final_model_{args.learning_rate}lr_{args.beta}beta")
    logger.info(f"Saving final model to {final_model_path}")
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)

    wandb.log({
        "train/final_loss":  train_result.metrics.get("train_loss", None),
        "train/total_steps": train_result.global_step,
    })
    wandb.finish()

    logger.info("✅ DPO training complete!")
    logger.info(f"   Model saved to: {final_model_path}")
    logger.info(f"   Pairs trained on: {len(train_dataset)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DPO Training for Multilingual Math Reasoning")
    parser.add_argument("--dataset_file",     type=str,   default=CONFIG["dataset_file"])
    parser.add_argument("--stats_file",       type=str,   default=CONFIG["stats_file"])
    parser.add_argument("--output_dir",       type=str,   default=CONFIG["output_dir"])
    parser.add_argument("--wandb_project",    type=str,   default=CONFIG["wandb_project"])
    parser.add_argument("--wandb_run_name",   type=str,   default=CONFIG["wandb_run_name"])
    parser.add_argument("--beta",             type=float, default=CONFIG["beta"])
    parser.add_argument("--learning_rate",    type=float, default=CONFIG["learning_rate"])
    parser.add_argument("--num_epochs",       type=int,   default=CONFIG["num_epochs"])
    parser.add_argument("--batch_size",       type=int,   default=CONFIG["batch_size"])
    parser.add_argument("--grad_accum_steps", type=int,   default=CONFIG["grad_accum_steps"])
    parser.add_argument("--max_length",       type=int,   default=CONFIG["max_length"])
    parser.add_argument("--max_pairs",        type=int,   default=CONFIG["max_pairs"])
    parser.add_argument("--cpu_test",         action="store_true", default=CONFIG["cpu_test"])
    parser.add_argument("--checkpoint_dir",   type=str,   default=None)
    args, _ = parser.parse_known_args()
    main(args)
