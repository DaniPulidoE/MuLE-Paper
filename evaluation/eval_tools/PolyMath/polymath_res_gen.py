import argparse
import json
import os

# Avoid FlashInfer JIT paths that require nvcc on shared cluster runtimes.
os.environ.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("VLLM_DISABLE_FLASHINFER_PREFILL", "1")

from datasets import load_dataset
from vllm import LLM, SamplingParams


LANG_TO_INSTRUCTIONS = {
    "en": "{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}.",
    "es": "{question}\nPor favor, razona paso a paso y pon tu respuesta final dentro de \\boxed{{}}.",
    "fr": "{question}\nVeuillez raisonner étape par étape et mettre votre réponse finale dans \\boxed{{}}.",
    "zh": "{question}\n请逐步推理，并将您的最终答案放在 \\boxed{{}} 中。",
    "ja": "{question}\nステップバイステップで推論し、最終的な答えを \\boxed{{}} の中に入れてください。",
    "th": "{question}\nกรุณาเหตุผลขั้นตอนต่อขั้นตอนและใส่คำตอบสุดท้ายของคุณใน \\boxed{{}}.",
    "ko": "{question}\n단계별로 추론하고 최종 답변을 \\boxed{{}} 안에 넣어주세요.",
    "pt": "{question}\nPor favor, raciocine passo a passo e coloque sua resposta final dentro de \\boxed{{}}.",
    "vi": "{question}\nVui lòng lý giải từng bước và đặt câu trả lời cuối cùng của bạn trong \\boxed{{}}.",
    "ar": "{question}\nيرجى المنطق خطوة بخطوة، ووضع إجابتك النهائية داخل \\boxed{{}}."
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", type=str, default="en", help="The language of the dataset.")
    parser.add_argument("--bench", type=str, default="eval_tools/PolyMath/data-polymath", help="The benchmark dataset.")
    parser.add_argument("--level", type=str, default="low", help="The level of the dataset.")
    parser.add_argument("--temp", type=float, default=0.6)
    parser.add_argument("--model_path", type=str, default="Models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    parser.add_argument("--model_name", type=str, default="DeepSeek-R1-Distill-Qwen-7B")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="The size of the tensor parallelism.")
    parser.add_argument("--max_model_len", type=int, default=36000)
    parser.add_argument("--max_tokens", type=int, default=32768)
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--max_num_seqs", type=int, default=None)
    parser.add_argument("--max_num_batched_tokens", type=int, default=None)
    parser.add_argument("--enforce_eager", action="store_true")
    parser.add_argument("--disable_prefix_caching", action="store_true")
    parser.add_argument("--inference_type", type=str, default="no_constrain")
    parser.add_argument("--save_path", type=str, default="logs-eval/PolyMath-temp_0.9")
    parser.add_argument(
        "--input_jsonl",
        type=str,
        default=None,
        help="Optional path to custom PolyMath-style JSONL (expects question, answer, language, source).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    sampling_params = SamplingParams(
        temperature=args.temp,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        n=args.num_samples,
    )
    llm_kwargs = {
        "model": args.model_path,
        "max_model_len": args.max_model_len,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enforce_eager": args.enforce_eager,
        "enable_prefix_caching": not args.disable_prefix_caching,
    }
    if args.max_num_seqs is not None:
        llm_kwargs["max_num_seqs"] = args.max_num_seqs
    if args.max_num_batched_tokens is not None:
        llm_kwargs["max_num_batched_tokens"] = args.max_num_batched_tokens

    llm = LLM(**llm_kwargs)
    if hasattr(llm, "get_tokenizer"):
        tokenizer = llm.get_tokenizer()
    else:
        tokenizer_obj = getattr(llm.llm_engine, "tokenizer", None)
        tokenizer = getattr(tokenizer_obj, "tokenizer", tokenizer_obj)

    print("Testing on language:", args.lang)
    if args.input_jsonl:
        src = []
        with open(args.input_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                if "question" not in item:
                    continue
                item_lang = item.get("language", "en")
                if item_lang != args.lang:
                    continue
                item_source = str(item.get("source", item.get("level", "")))
                if item_source and item_source != args.level:
                    continue
                if "answer" not in item and "ground_truth" in item:
                    item["answer"] = str(item["ground_truth"])
                if "answer" not in item:
                    continue
                src.append(item)
    else:
        src = load_dataset(f"{args.bench}/{args.lang}")["train"]

    all_prompts = []
    inputs = []
    for item in src:
        item_id = str(item.get("id", ""))
        item_level = str(item.get("source", item.get("level", "")))
        if (args.input_jsonl and ((not item_level) or item_level == args.level)) or (
            (not args.input_jsonl) and (args.level in item_id)
        ):
            if args.inference_type != "no_constrain":
                raise ValueError(f"Unsupported inference_type: {args.inference_type}")

            formatted_prompt = LANG_TO_INSTRUCTIONS[args.lang].format(question=item["question"])
            chat_template_prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": formatted_prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            all_prompts.append(chat_template_prompt)
            inputs.append(item)

    print(len(all_prompts))
    outputs = llm.generate(all_prompts, sampling_params)

    save_res = []
    for output, item in zip(outputs, inputs):
        for i, output_i in enumerate(output.outputs):
            generated_text = output_i.text
            if "</think>" in generated_text:
                tmp = generated_text.split("</think>")
                if len(tmp) == 2:
                    item[f"thinking_pred_{i}"] = tmp[0]
                    item[f"answer_pred_{i}"] = tmp[1]
                else:
                    item[f"thinking_pred_{i}"] = None
                    item[f"answer_pred_{i}"] = generated_text
            else:
                item[f"thinking_pred_{i}"] = None
                item[f"answer_pred_{i}"] = generated_text
        save_res.append(item)

    print(len(save_res))

    cur_save_path = f"{args.save_path}/{args.model_name}/{args.level}"
    os.makedirs(cur_save_path, exist_ok=True)
    with open(f"{cur_save_path}/{args.lang}.json", "w", encoding="utf-8") as f:
        json.dump(save_res, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
