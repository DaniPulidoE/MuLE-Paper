import json
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from transformers import AutoTokenizer
import tqdm

# Import your custom accuracy scoring function
from rewards import acc_compute_score

# Load langdetect
try:
    from langdetect import detect_langs, DetectorFactory, LangDetectException
    DetectorFactory.seed = 42
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False
    logging.warning("langdetect not installed. Language consistency check will fail.")

CANDIDATES_FILE = "./Datos/iter1_questions.jsonl"
MODEL_NAME = "XueZhang-bjtu/1.5B-cold-start-SFT"

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

def format_reward(text):
    """Checks if the output contains <think> tags and a prompt \boxed{} answer."""
    if not re.search(r'<think>', text):
        return 0
    if not re.search(r'</think>', text):
        return 0

    close_pos = text.rfind('</think>')
    boxes_after = [m.start() for m in re.finditer(r'\\boxed\{', text)
                   if m.start() > close_pos]
    if not boxes_after:
        return 0

    # \boxed{} appears promptly after </think>
    if (min(boxes_after) - close_pos) > 800:
        return 0

    return 1

def language_consistency_score(text: str, expected_lang: str) -> float:
    """Returns a continuous language consistency score in [0.0, 1.0]."""
    if not LANGDETECT_AVAILABLE:
        return 1.0

    text_for_detection = re.sub(r'\\boxed\{[^}]*\}', '', text)
    text_for_detection = re.sub(r'\$[^$]*\$', '', text_for_detection)
    text_for_detection = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', '', text_for_detection)
    text_for_detection = re.sub(r'<think>|</think>', '', text_for_detection)
    text_for_detection = text_for_detection.strip()

    if len(text_for_detection) < 20:
        return 1.0

    try:
        lang_probs = detect_langs(text_for_detection)
        for lp in lang_probs:
            if lp.lang == expected_lang:
                return float(lp.prob)
        return 0.0
    except LangDetectException:
        return 0.0

def print_stats_helper(df_subset, column, label, is_binary=False):
    """Helper to print summary statistics for various metrics"""
    print("\n" + "-"*40)
    print(f"      {column.upper()} STATS FOR: {label}")
    print("-"*40)
    
    if df_subset.empty:
        print("No data available for this subset.")
        return

    total_points = len(df_subset)
    print(f"Total answers: {total_points}")
    
    if is_binary:
        correct_count = df_subset[column].sum()
        pct = (correct_count / total_points) * 100
        print(f"Correct Format: {correct_count} ({pct:.1f}%)")
    else:
        print(f"Average {column}: {df_subset[column].mean():.2f}")
        print(f"Quantiles:")
        quantiles = df_subset[column].quantile([0.0, 0.25, 0.5, 0.75, 1.0])
        print(f"  Min (0%):   {quantiles[0.0]:.2f}")
        print(f"  Q1  (25%):  {quantiles[0.25]:.2f}")
        print(f"  Q2  (50%):  {quantiles[0.50]:.2f} (Median)")
        print(f"  Q3  (75%):  {quantiles[0.75]:.2f}")
        print(f"  Max (100%): {quantiles[1.0]:.2f}")

def main():
    if not LANGDETECT_AVAILABLE:
        print("Please install langdetect first: pip install langdetect")
        return

    print("Loading Tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return

    data_records = []
    
    print("Reading file and calculating all scores (Acc, LC, Backtracks, Length, Format)...")
    with open(CANDIDATES_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            
            try:
                item = json.loads(line)
                expected_lang = item.get("language")
                ground_truth = item.get("ground_truth")
                candidates = item.get("candidates", [])
                
                if ground_truth is None:
                    continue
                
                for answer in candidates:
                    # Calculate Metrics
                    acc = acc_compute_score(answer, ground_truth)
                    lc_score = language_consistency_score(answer, expected_lang)
                    bt_count = len(BACKTRACK_SIGNALS.findall(answer))
                    ans_length = len(tokenizer.encode(answer))
                    fmt_score = format_reward(answer)
                    
                    data_records.append({
                        "Language": expected_lang,
                        "LC Score": lc_score,
                        "Backtracks": bt_count,
                        "Length": ans_length,
                        "Format": fmt_score,
                        "Accuracy": acc
                    })
            except json.JSONDecodeError as e:
                print(f"Skipping line due to JSON error: {e}")
                continue

    if not data_records:
        print("No valid data found.")
        return

    # Convert to pandas DataFrame
    df = pd.DataFrame(data_records)
    df["Accuracy Label"] = df["Accuracy"].map({0: "Incorrect (Acc=0)", 1: "Correct (Acc=1)"})

    # Subsets for splitting metrics
    df_acc0 = df[df["Accuracy"] == 0]
    df_acc1 = df[df["Accuracy"] == 1]

    # --- Print Global Statistics ---
    print("\n" + "="*50)
    print("      GLOBAL FORMAT COMPLIANCE SUMMARY")
    print("="*50)
    print_stats_helper(df, "Format", "OVERALL (All Answers)", is_binary=True)
    print_stats_helper(df_acc0, "Format", "INCORRECT ANSWERS (Acc = 0)", is_binary=True)
    print_stats_helper(df_acc1, "Format", "CORRECT ANSWERS (Acc = 1)", is_binary=True)

    print("\n" + "="*50)
    print("      GLOBAL ACCURACY SUMMARY")
    print("="*50)
    total_answers = len(df)
    correct_answers = df["Accuracy"].sum()
    wrong_answers = total_answers - correct_answers
    print(f"Total Answers: {total_answers}")
    print(f"Correct (Acc=1): {correct_answers} ({(correct_answers/total_answers)*100:.1f}%)")
    print(f"Wrong (Acc=0): {wrong_answers} ({(wrong_answers/total_answers)*100:.1f}%)")

    # --- NEW: Breakdown by Language ---
    print("\n" + "="*80)
    print("      SUMMARY BY LANGUAGE (ALL METRICS)")
    print("="*80)
    
    # Grouping by Language to get overall language stats
    lang_summary = df.groupby('Language').agg(
        Total_Ans=('Accuracy', 'count'),
        Correct_Ans=('Accuracy', 'sum'),
        Acc_Pct=('Accuracy', lambda x: x.mean() * 100),
        Format_Pct=('Format', lambda x: x.mean() * 100),
        Avg_LC=('LC Score', 'mean'),
        Avg_BTs=('Backtracks', 'mean'),
        Avg_Len=('Length', 'mean')
    ).reset_index()
    
    # Format the float columns for cleaner display
    format_mapping = {'Acc_Pct': '{:.1f}%', 'Format_Pct': '{:.1f}%', 'Avg_LC': '{:.2f}', 'Avg_BTs': '{:.2f}', 'Avg_Len': '{:.0f}'}
    for col, fmt in format_mapping.items():
        lang_summary[col] = lang_summary[col].map(fmt.format)
    
    print(lang_summary.to_string(index=False))

    # --- NEW: Breakdown by Language AND Accuracy ---
    print("\n" + "="*90)
    print("      SUMMARY BY LANGUAGE AND ACCURACY (ALL METRICS)")
    print("="*90)
    
    # Grouping by Language and Accuracy Label
    lang_acc_summary = df.groupby(['Language', 'Accuracy Label']).agg(
        Total_Ans=('Accuracy', 'count'),
        Format_Pct=('Format', lambda x: x.mean() * 100),
        Avg_LC=('LC Score', 'mean'),
        Avg_BTs=('Backtracks', 'mean'),
        Avg_Len=('Length', 'mean')
    ).reset_index()
    
    # Format the float columns for cleaner display
    format_mapping_acc = {'Format_Pct': '{:.1f}%', 'Avg_LC': '{:.2f}', 'Avg_BTs': '{:.2f}', 'Avg_Len': '{:.0f}'}
    for col, fmt in format_mapping_acc.items():
        lang_acc_summary[col] = lang_acc_summary[col].map(fmt.format)
    
    print(lang_acc_summary.to_string(index=False))


    # --- Visualizations ---
    
    # 1. Format Compliance Plot (Bar Plot of Percentages)
    print("\nGenerating Bar Plot for Format Compliance Percentages...")
    plt.figure(figsize=(12, 7))
    
    # We use a barplot for binary data because the mean of a 0/1 variable IS the percentage.
    df["Format Percentage"] = df["Format"] * 100
    
    sns.barplot(
        data=df, x="Language", y="Format Percentage", hue="Accuracy Label",
        palette={"Incorrect (Acc=0)": "#ff9999", "Correct (Acc=1)": "#66b3ff"},
        edgecolor="black", alpha=0.9, capsize=0.1
    )
    
    plt.title("Percentage of Answers with Correct Formatting", fontsize=15, pad=15)
    plt.xlabel("Expected Language", fontsize=12)
    plt.ylabel("% of Answers with Correct Format", fontsize=12)
    plt.ylim(0, 105) 
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.legend(title="Accuracy Status", loc="lower right", framealpha=0.9)
    
    output_image_fmt = "./Plots/format_compliance_by_acc_barplot.png"
    plt.savefig(output_image_fmt, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {output_image_fmt}")
    plt.close()

    # 2. Language Consistency Boxplot
    print("Generating Boxplot/Stripplot for Language Consistency...")
    plt.figure(figsize=(12, 7))
    sns.boxplot(
        data=df, x="Language", y="LC Score", hue="Accuracy Label",
        palette={"Incorrect (Acc=0)": "#ff9999", "Correct (Acc=1)": "#66b3ff"},
        showmeans=True, meanprops={"marker":"o", "markerfacecolor":"white", "markeredgecolor":"black"},
        fliersize=0, boxprops={"alpha": 0.7}
    )
    sns.stripplot(
        data=df, x="Language", y="LC Score", hue="Accuracy Label",
        palette={"Incorrect (Acc=0)": "#e63946", "Correct (Acc=1)": "#1d3557"},
        dodge=True, jitter=0.25, alpha=0.4, linewidth=0, size=4, legend=False
    )
    plt.title("Language Consistency Scores by Language and Accuracy", fontsize=15, pad=15)
    plt.xlabel("Expected Language", fontsize=12)
    plt.ylabel("Language Consistency Score (0.0 to 1.0)", fontsize=12)
    plt.ylim(-0.05, 1.05) 
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.legend(title="Accuracy Status", loc="lower right", framealpha=0.9)
    plt.savefig("./Plots/language_consistency_by_acc_boxplot.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 3. Backtracks Plot
    print("Generating Boxplot/Stripplot for Backtracks...")
    plt.figure(figsize=(12, 7))
    sns.boxplot(
        data=df, x="Language", y="Backtracks", hue="Accuracy Label",
        palette={"Incorrect (Acc=0)": "#ff9999", "Correct (Acc=1)": "#66b3ff"},
        showmeans=True, meanprops={"marker":"o", "markerfacecolor":"white", "markeredgecolor":"black"},
        fliersize=0, boxprops={"alpha": 0.7}
    )
    sns.stripplot(
        data=df, x="Language", y="Backtracks", hue="Accuracy Label",
        palette={"Incorrect (Acc=0)": "#e63946", "Correct (Acc=1)": "#1d3557"},
        dodge=True, jitter=0.25, alpha=0.4, linewidth=0, size=4, legend=False
    )
    plt.title("Number of Backtracks by Language and Accuracy", fontsize=15, pad=15)
    plt.xlabel("Expected Language", fontsize=12)
    plt.ylabel("Total Backtracks", fontsize=12)
    plt.ylim(-0.5, df["Backtracks"].max() + 1.5)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.legend(title="Accuracy Status", loc="upper right", framealpha=0.9)
    plt.savefig("./Plots/backtracks_by_acc_boxplot.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 4. Token Length Plot
    print("Generating Boxplot/Stripplot for Answer Token Length...")
    plt.figure(figsize=(12, 7))
    sns.boxplot(
        data=df, x="Language", y="Length", hue="Accuracy Label",
        palette={"Incorrect (Acc=0)": "#ff9999", "Correct (Acc=1)": "#66b3ff"},
        showmeans=True, meanprops={"marker":"o", "markerfacecolor":"white", "markeredgecolor":"black"},
        fliersize=0, boxprops={"alpha": 0.7}
    )
    sns.stripplot(
        data=df, x="Language", y="Length", hue="Accuracy Label",
        palette={"Incorrect (Acc=0)": "#e63946", "Correct (Acc=1)": "#1d3557"},
        dodge=True, jitter=0.25, alpha=0.4, linewidth=0, size=3, legend=False
    )
    plt.title("Answer Length (Tokens) by Language and Accuracy", fontsize=15, pad=15)
    plt.xlabel("Expected Language", fontsize=12)
    plt.ylabel("Length of Answer (Tokens)", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.legend(title="Accuracy Status", loc="upper right", framealpha=0.9)
    plt.savefig("./Plots/length_by_acc_boxplot.png", dpi=300, bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    main()