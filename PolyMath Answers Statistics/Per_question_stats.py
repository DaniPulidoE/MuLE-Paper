import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer

from polymath_data import MODEL_NAME, LEVELS, NUM_SAMPLES, load_records


def main():
    print("Loading Tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return

    print("Reading PolyMath generations and processing metrics per question difficulty...")
    records = load_records(tokenizer=tokenizer)
    if not records:
        print("No valid data found.")
        return

    df = pd.DataFrame(records)
    df["Level"] = pd.Categorical(df["Level"], categories=LEVELS, ordered=True)

    # One row per unique question, for the distribution plot.
    df_questions = df.drop_duplicates(subset=["Level", "Language", "Question ID"])[
        ["Level", "Language", "Question Correct Count"]
    ]

    # --- Print Statistics ---
    print("\n" + "="*60)
    print("   AVERAGE METRICS BY QUESTION SCORE, PER DIFFICULTY LEVEL")
    print("="*60)

    for level in LEVELS:
        print(f"\n--- {level.upper()} ---")
        df_level = df[df["Level"] == level]
        grouped = df_level.groupby("Question Correct Count")

        print(f"{'Q-Score':<10} | {'Avg Tokens':<15} | {'Avg Backtracks':<15} | {'Total Answers in Bin'}")
        print("-" * 65)

        for score in range(NUM_SAMPLES + 1):
            if score in grouped.groups:
                group_data = grouped.get_group(score)
                avg_tokens = group_data["Length"].mean()
                avg_bts = group_data["Backtracks"].mean()
                num_answers = len(group_data)
                print(f"{score:<10} | {avg_tokens:<15.1f} | {avg_bts:<15.2f} | {num_answers}")
            else:
                print(f"{score:<10} | {'N/A':<15} | {'N/A':<15} | 0")

    # --- Visualizations ---
    print("\nGenerating trend plots...")
    os.makedirs("./Plots", exist_ok=True)

    # 1. Backtracks vs. Question Score, split by Level
    plt.figure(figsize=(10, 6))
    sns.pointplot(
        data=df,
        x="Question Correct Count",
        y="Backtracks",
        hue="Level",
        palette="Set2",
        capsize=0.1,
        errwidth=1.5
    )
    plt.title("Average Backtracks vs. Per-Question Difficulty, by Level", fontsize=15, pad=15)
    plt.xlabel(f"Number of Correct Samples for the Question (0=Hardest, {NUM_SAMPLES}=Easiest)", fontsize=12)
    plt.ylabel("Average Number of Backtracks", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(title="Level")

    output_image_bt_trend = "./Plots/trend_backtracks_by_qscore.png"
    plt.savefig(output_image_bt_trend, dpi=300, bbox_inches="tight")
    plt.close()

    # 2. Token Length vs. Question Score, split by Level
    plt.figure(figsize=(10, 6))
    sns.pointplot(
        data=df,
        x="Question Correct Count",
        y="Length",
        hue="Level",
        palette="Set2",
        capsize=0.1,
        errwidth=1.5
    )
    plt.title("Average Answer Length (Tokens) vs. Per-Question Difficulty, by Level", fontsize=15, pad=15)
    plt.xlabel(f"Number of Correct Samples for the Question (0=Hardest, {NUM_SAMPLES}=Easiest)", fontsize=12)
    plt.ylabel("Average Answer Length (Tokens)", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(title="Level")

    output_image_len_trend = "./Plots/trend_tokens_by_qscore.png"
    plt.savefig(output_image_len_trend, dpi=300, bbox_inches="tight")
    plt.close()

    # 3. Distribution of correct samples per question, split by Level
    plt.figure(figsize=(12, 7))
    ax = sns.countplot(
        data=df_questions,
        x="Question Correct Count",
        hue="Level",
        palette="Set2",
        edgecolor="black",
        alpha=0.9
    )
    plt.title("Distribution of Correct Samples per Question, by Level", fontsize=15, pad=15)
    plt.xlabel(f"Number of Correct Samples (out of {NUM_SAMPLES})", fontsize=12)
    plt.ylabel("Number of Questions", fontsize=12)

    ax.set_xticks(range(NUM_SAMPLES + 1))
    ax.set_xticklabels(range(NUM_SAMPLES + 1))

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(title="Level", loc="upper right")

    output_image_dist = "./Plots/correct_answers_per_question_barplot.png"
    plt.savefig(output_image_dist, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Plots saved to:\n - {output_image_bt_trend}\n - {output_image_len_trend}\n - {output_image_dist}")


if __name__ == "__main__":
    main()