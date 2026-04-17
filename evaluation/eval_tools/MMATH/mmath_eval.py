import argparse
import json
import os
import time
from collections import defaultdict

# Avoid FlashInfer JIT paths that require nvcc on shared cluster runtimes.
os.environ.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("VLLM_DISABLE_FLASHINFER_PREFILL", "1")
os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"

from math_verify import parse, verify
from tqdm import tqdm
from vllm import LLM, SamplingParams

from utils import math_postprocess_v2


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
    parser.add_argument("--model_path", type=str, default="Models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    parser.add_argument("--model_name", type=str, default="DeepSeek-R1-Distill-Qwen-7B")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="The size of the tensor parallelism.")
    parser.add_argument("--lang", type=str, default="all", help="The language of the dataset.", nargs="+")
    parser.add_argument("--bench", type=str, default="mmath", help="The benchmark dataset.")
    parser.add_argument("--temp", type=float, default=0.6)
    parser.add_argument("--max_model_len", type=int, default=36000)
    parser.add_argument("--max_tokens", type=int, default=32768)
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--max_num_seqs", type=int, default=None)
    parser.add_argument("--max_num_batched_tokens", type=int, default=None)
    parser.add_argument("--enforce_eager", action="store_true")
    parser.add_argument("--disable_prefix_caching", action="store_true")
    parser.add_argument(
        "--log_lang_timing",
        action="store_true",
        help="Run generation per-language and print elapsed time for each language in logs.",
    )
    parser.add_argument(
        "--input_jsonl",
        type=str,
        default=None,
        help="Optional path to custom MMATH-style JSONL (expects question, answer, language).",
    )
    return parser.parse_args()


def save_results(mmath, model_name, temp, lang):
    os.makedirs(f"logs-eval/MMATH-temp_{temp}/{model_name}", exist_ok=True)
    with open(f"logs-eval/MMATH-temp_{temp}/{model_name}/{lang}.json", "w", encoding="utf-8") as f:
        json.dump(mmath[lang], f, ensure_ascii=False, indent=4)


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

    default_languages = ["en", "zh", "ar", "es", "fr", "ja", "ko", "pt", "th", "vi"]
    custom_rows_by_lang = None
    if args.input_jsonl:
        custom_rows_by_lang = defaultdict(list)
        with open(args.input_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                if "question" not in item:
                    continue

                lang = item.get("language", "en")
                if lang not in LANG_TO_INSTRUCTIONS:
                    lang = "en"

                answer = item.get("answer", item.get("ground_truth", None))
                if answer is None:
                    continue

                data_source = item.get("data_source", item.get("source", "MATH500"))
                if data_source not in {"AIME2024", "AIME2025", "CNMO", "MATH500"}:
                    data_source = "MATH500"

                custom_rows_by_lang[lang].append(
                    {
                        "question": item["question"],
                        "answer": str(answer),
                        "data_source": data_source,
                        "id": item.get("id"),
                        "language": lang,
                    }
                )

    if args.lang != "all":
        languages = args.lang
    elif custom_rows_by_lang is not None:
        languages = [lang for lang in default_languages if lang in custom_rows_by_lang]
    else:
        languages = default_languages
    print("Testing on languages:", languages)

    mmath = {}
    all_prompts = []
    prompt_lang_idx = []

    for lang in languages:
        if custom_rows_by_lang is not None:
            mmath[lang] = custom_rows_by_lang.get(lang, [])
        else:
            with open(f"eval_tools/MMATH/{args.bench}/{lang}.json", "r", encoding="utf-8") as f:
                mmath[lang] = json.load(f)

        for i, item in enumerate(mmath[lang]):
            formatted_prompt = LANG_TO_INSTRUCTIONS[lang].format(question=item["question"])
            chat_template_prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": formatted_prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )

            mmath[lang][i]["final_prompt"] = chat_template_prompt
            all_prompts.append(chat_template_prompt)
            prompt_lang_idx.append((lang, i))

    def postprocess_one_output(lang, idx, output):
        for i, output_i in enumerate(output.outputs):
            generated_text = output_i.text
            mmath[lang][idx][f"prediction_{i}"] = generated_text
            mmath[lang][idx][f"pred_answer_{i}"] = math_postprocess_v2(generated_text)

            if mmath[lang][idx][f"pred_answer_{i}"] is None:
                if_correct = False
            else:
                gold = parse(mmath[lang][idx]["answer"])
                pred = parse("$" + mmath[lang][idx][f"pred_answer_{i}"] + "$")
                if_correct = verify(gold, pred)

            mmath[lang][idx][f"correct_{i}"] = if_correct

    if args.log_lang_timing:
        for lang in languages:
            lang_prompts = [mmath[lang][i]["final_prompt"] for i in range(len(mmath[lang]))]
            lang_indices = [(lang, i) for i in range(len(mmath[lang]))]
            if len(lang_prompts) == 0:
                continue

            t0 = time.perf_counter()
            lang_outputs = llm.generate(lang_prompts, sampling_params)
            elapsed = time.perf_counter() - t0
            qps = len(lang_prompts) / elapsed if elapsed > 0 else 0.0
            print(
                f"[TIMING] lang={lang} prompts={len(lang_prompts)} "
                f"elapsed_s={elapsed:.2f} qps={qps:.3f}"
            )

            for output, (_, idx) in tqdm(
                zip(lang_outputs, lang_indices),
                total=len(lang_indices),
                desc=f"postprocess-{lang}",
            ):
                postprocess_one_output(lang, idx, output)
    else:
        t0 = time.perf_counter()
        outputs = llm.generate(all_prompts, sampling_params)
        elapsed = time.perf_counter() - t0
        qps = len(all_prompts) / elapsed if elapsed > 0 else 0.0
        print(
            f"[TIMING] lang=all prompts={len(all_prompts)} "
            f"elapsed_s={elapsed:.2f} qps={qps:.3f}"
        )

        for output, (lang, idx) in tqdm(zip(outputs, prompt_lang_idx), total=len(prompt_lang_idx)):
            postprocess_one_output(lang, idx, output)

    for lang in languages:
        save_results(mmath, args.model_name, args.temp, lang)


if __name__ == "__main__":
    main()
