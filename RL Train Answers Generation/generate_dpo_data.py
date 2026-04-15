import time
import json
import os
import argparse
from vllm import LLM, SamplingParams
import torch

LANG_TO_INSTRUCTIONS = {
    'en': "{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}.",
    'es': "{question}\nPor favor, razona paso a paso y pon tu respuesta final dentro de \\boxed{{}}.",
    'fr': "{question}\nVeuillez raisonner étape par étape et mettre votre réponse finale dans \\boxed{{}}.",
    'zh': "{question}\n请逐步推理，并将您的最终答案放在 \\boxed{{}} 中。",
    'ja': "{question}\nステップバイステップで推論し、最終的な答えを \\boxed{{}} の中に入れてください。",
    'th': "{question}\nกรุณาเหตุผลขั้นตอนต่อขั้นตอนและใส่คำตอบสุดท้ายของคุณใน \\boxed{{}}.",
    'ko': "{question}\n단계별로 추론하고 최종 답변을 \\boxed{{}} 안에 넣어주세요.",
    'pt': "{question}\nPor favor, raciocine passo a passo e coloque sua resposta final dentro de \\boxed{{}}.",
    'vi': "{question}\nVui lòng lý giải từng bước và đặt câu trả lời cuối cùng của bạn trong \\boxed{{}}.",
    'ar': "{question}\nيرجى المنطق خطوة بخطوة، ووضع إجابتك النهائية داخل \\boxed{{}}."
}

def main(args):
    # --- SETUP PATHS ---
    BASE_DIR = args.base_dir
    INPUT_FILE = os.path.join(BASE_DIR, args.input_file)
    OUTPUT_FILE = os.path.join(BASE_DIR, args.output_file)
    
    MODEL_NAME = "XueZhang-bjtu/1.5B-cold-start-SFT"
    CHUNK_SIZE = 50  # How many questions to process before saving to disk

    # --- GPU VERIFICATION CHECK ---
    print("\n" + "="*40)
    print("🔍 GPU VERIFICATION CHECK")
    print("="*40)
    print(f"CUDA_VISIBLE_DEVICES env var: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs PyTorch can see: {num_gpus}")

    if num_gpus > 0:
        current_device = torch.cuda.current_device()
        print(f"PyTorch Internal Device ID: {current_device}")
        print(f"GPU Name: {torch.cuda.get_device_name(current_device)}")
        
        vram_gb = torch.cuda.get_device_properties(current_device).total_memory / 1e9
        print(f"Total VRAM Available: {vram_gb:.2f} GB")
    else:
        print("❌ ERROR: PyTorch cannot see any GPUs!")
    print("="*40 + "\n")

    # --- 1. RESUMPTION LOGIC ---
    processed_ids = set()
    if os.path.exists(OUTPUT_FILE):
        print(f"📄 Found existing output file. Scanning for completed questions...")
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    processed_ids.add(item["id"])
        print(f"⏭️ Skipping {len(processed_ids)} questions that are already completed.")

    # --- 2. LOAD AND FILTER INPUT ---
    print(f"Loading questions from {INPUT_FILE}...")
    pending_data = []
    
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                # Only add if we haven't already processed it
                if item["id"] not in processed_ids:
                    pending_data.append(item)
                    
    if not pending_data:
        print("🎉 All questions in the input file have already been processed!")
        return

    print(f"⏳ {len(pending_data)} questions remaining to process.")

    # --- 3. INITIALIZE vLLM ---
    print(f"\nLoading model {MODEL_NAME} onto the RTX 6000...")
    llm = LLM(
        model=MODEL_NAME,
        dtype="half",
        gpu_memory_utilization=0.75, 
        max_model_len=16384       
    )

    # --- 4. CONFIGURE SAMPLING ---
    sampling_params = SamplingParams(
        n=8,                     
        temperature=0.9,         
        top_p=0.95,              
        max_tokens=14336          
    )

    # --- 5. CHUNKED GENERATION ---
    print(f"\n🚀 Starting chunked generation (Chunk size: {CHUNK_SIZE})...")
    start_time = time.perf_counter()

    for i in range(0, len(pending_data), CHUNK_SIZE):
        chunk_data = pending_data[i:i + CHUNK_SIZE]

        prompts = []
        for item in chunk_data:
            lang = item.get('lan', 'en')
            if lang not in LANG_TO_INSTRUCTIONS:
                lang = 'en'

            question_text = LANG_TO_INSTRUCTIONS[lang].format(question=item['question'])
            
            messages = [{"role": "user", "content": question_text}]
            formatted_string = llm.get_tokenizer().apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True,
                enable_thinking=True
            )
            prompts.append(formatted_string)
        
        print(f"\nProcessing chunk {i // CHUNK_SIZE + 1} (Questions {i+1} to {min(i+CHUNK_SIZE, len(pending_data))})...")
        
        # Generate answers for just this chunk
        outputs = llm.generate(prompts, sampling_params)

        # Append the results to the JSONL file immediately
        with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
            for j, output in enumerate(outputs):
                original_item = chunk_data[j]
                
                # Extract the generated candidate texts
                candidates = ["<think>\n" + generated_sequence.text for generated_sequence in output.outputs]
                
                # Create the result dictionary with the new fields
                result_dict = {
                    "id": original_item["id"],
                    "numerical_id": original_item.get("num_id"),
                    "question": original_item["question"],
                    "ground_truth": original_item.get("answer"),
                    "language": original_item.get("lan", "en"),
                    "candidates": candidates
                }
                
                # Write as a single JSON line
                f.write(json.dumps(result_dict, ensure_ascii=False) + "\n")
                
        print(f"💾 Saved {len(chunk_data)} questions to disk.")
        #f.flush()  # Ensure data is written to disk

    end_time = time.perf_counter()
    elapsed_seconds = end_time - start_time
    elapsed_minutes = elapsed_seconds / 60

    print("\n" + "="*40)
    print(f"✅ Generation Complete!")
    print(f"⏱️ Total Elapsed Time: {elapsed_minutes:.2f} minutes ({elapsed_seconds:.2f} seconds)")
    print("="*40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate DPO Candidate Answers")
    
    # Define the required arguments
    parser.add_argument("--base_dir", type=str, required=True, help="Base directory path")
    parser.add_argument("--input_file", type=str, required=True, help="Input JSONL file name")
    parser.add_argument("--output_file", type=str, required=True, help="Output JSONL file name")
    
    args = parser.parse_args()
    main(args)