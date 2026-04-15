import json
import re
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer

# Import your custom accuracy scoring function
from rewards import acc_compute_score

CANDIDATES_FILE = "./Datos/iter1_questions.jsonl"
MODEL_NAME = "XueZhang-bjtu/1.5B-cold-start-SFT"
N_CANDIDATES = 8

# Regex for detecting backtrack signals
BACKTRACK_SIGNALS = re.compile(
    r'\bwait\b|\bactually\b|\bno,?\s+wait\b|'
    r'\blet me reconsider\b|\blet me restart\b|'
    r"that'?s wrong\b|\bi made an error\b|"
    r'\blet me try again\b|\bhmm,?\s+actually\b|'
    r'\bhold on\b|\bon second thought\b|'
    # French
    r'\battends\b|\ben fait\b|\bnon,?\s+attendez\b|'
    r'\bje me suis tromp|\breprenons\b|'
    # Portuguese
    r'\bespera\b|\bna verdade\b|\bnão,?\s+espera\b|'
    r'\bcometi um erro\b',
    re.IGNORECASE
)

def main():
    print("Loading Tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return

    data_records = []
    question_records = []  # NEW: to track data exactly once per question
    
    print("Reading file and processing metrics per question difficulty...")
    with open(CANDIDATES_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            
            try:
                item = json.loads(line)
                expected_lang = item.get("language", "unknown")
                ground_truth = item.get("ground_truth")
                candidates = item.get("candidates", [])
                
                if ground_truth is None or not candidates:
                    continue
                
                # 1. Determine the question's overall "score" (0 to 8 correct)
                correct_count = sum(1 for ans in candidates if acc_compute_score(ans, ground_truth) == 1)
                
                # 2. Save the question-level record (happens ONCE per question)
                question_records.append({
                    "Language": expected_lang,
                    "Question Score (0-8)": correct_count
                })
                
                # 3. Calculate metrics for ALL candidates generated for this question (8 times per question)
                for answer in candidates:
                    ans_length = len(tokenizer.encode(answer))
                    bt_count = len(BACKTRACK_SIGNALS.findall(answer))
                    
                    data_records.append({
                        "Language": expected_lang,
                        "Question Score (0-8)": correct_count,
                        "Tokens": ans_length,
                        "Backtracks": bt_count
                    })
                    
            except json.JSONDecodeError as e:
                print(f"Skipping line due to JSON error: {e}")
                continue

    if not data_records:
        print("No valid data found.")
        return

    # Convert to pandas DataFrames
    df = pd.DataFrame(data_records)
    df_questions = pd.DataFrame(question_records)

    # --- Print Statistics ---
    print("\n" + "="*60)
    print("   AVERAGE METRICS BY QUESTION SCORE AND LANGUAGE")
    print("="*60)
    
    langs = sorted(df["Language"].unique())
    
    for lang in langs:
        print(f"\n--- {lang.upper()} ---")
        df_lang = df[df["Language"] == lang]
        
        # Group by the 0-8 score
        grouped = df_lang.groupby("Question Score (0-8)")
        
        # Print a formatted table header
        print(f"{'Q-Score':<10} | {'Avg Tokens':<15} | {'Avg Backtracks':<15} | {'Total Answers in Bin'}")
        print("-" * 65)
        
        for score in range(N_CANDIDATES + 1):
            if score in grouped.groups:
                group_data = grouped.get_group(score)
                avg_tokens = group_data["Tokens"].mean()
                avg_bts = group_data["Backtracks"].mean()
                num_answers = len(group_data)
                
                print(f"{score:<10} | {avg_tokens:<15.1f} | {avg_bts:<15.2f} | {num_answers}")
            else:
                print(f"{score:<10} | {'N/A':<15} | {'N/A':<15} | 0")

    # --- Visualizations ---
    print("\nGenerating trend plots...")
    os.makedirs("./Plots", exist_ok=True)  # Make sure the directory exists
    
    # 1. Backtracks vs. Question Score Plot
    plt.figure(figsize=(10, 6))
    sns.pointplot(
        data=df, 
        x="Question Score (0-8)", 
        y="Backtracks", 
        hue="Language", 
        palette="Set2",
        markers=["o", "s", "D"], 
        capsize=0.1,      # Adds error bar caps
        errwidth=1.5      # Error bar thickness
    )
    plt.title("Average Backtracks vs. Question Difficulty", fontsize=15, pad=15)
    plt.xlabel("Number of Correct Answers for the Question (0=Hardest, 8=Easiest)", fontsize=12)
    plt.ylabel("Average Number of Backtracks", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(title="Language")
    
    output_image_bt_trend = "./Plots/trend_backtracks_by_qscore.png"
    plt.savefig(output_image_bt_trend, dpi=300, bbox_inches="tight")
    plt.close()

    # 2. Token Length vs. Question Score Plot
    plt.figure(figsize=(10, 6))
    sns.pointplot(
        data=df, 
        x="Question Score (0-8)", 
        y="Tokens", 
        hue="Language", 
        palette="Set2",
        markers=["o", "s", "D"], 
        capsize=0.1, 
        errwidth=1.5
    )
    plt.title("Average Answer Length (Tokens) vs. Question Difficulty", fontsize=15, pad=15)
    plt.xlabel("Number of Correct Answers for the Question (0=Hardest, 8=Easiest)", fontsize=12)
    plt.ylabel("Average Answer Length (Tokens)", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(title="Language")
    
    output_image_len_trend = "./Plots/trend_tokens_by_qscore.png"
    plt.savefig(output_image_len_trend, dpi=300, bbox_inches="tight")
    plt.close()
    
    # 3. NEW: Number of Questions per Score Plot (Distribution)
    plt.figure(figsize=(12, 7))
    ax = sns.countplot(
        data=df_questions, 
        x="Question Score (0-8)", 
        hue="Language", 
        palette="Set2",
        edgecolor="black",
        alpha=0.9
    )
    plt.title("Distribution of Correct Answers per Question", fontsize=15, pad=15)
    plt.xlabel("Number of Correct Candidates (out of 8)", fontsize=12)
    plt.ylabel("Number of Questions", fontsize=12)
    
    # Ensure all x-ticks from 0 to 8 are shown even if a category is empty
    ax.set_xticks(range(N_CANDIDATES + 1))
    ax.set_xticklabels(range(N_CANDIDATES + 1))
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(title="Language", loc="upper right")
    
    output_image_dist = "./Plots/correct_answers_per_question_barplot.png"
    plt.savefig(output_image_dist, dpi=300, bbox_inches="tight")
    
    print(f"Plots saved to:\n - {output_image_bt_trend}\n - {output_image_len_trend}\n - {output_image_dist}")
    plt.show()

if __name__ == "__main__":
    main()