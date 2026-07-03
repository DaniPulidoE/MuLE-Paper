import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer

from polymath_data import MODEL_NAME, LEVELS, LANGDETECT_AVAILABLE, load_records


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

    print("Reading PolyMath generations and calculating all scores (Acc, LC, Backtracks, Length, Format)...")
    records = load_records(tokenizer=tokenizer)
    if not records:
        print("No valid data found.")
        return

    df = pd.DataFrame(records)
    df["Level"] = pd.Categorical(df["Level"], categories=LEVELS, ordered=True)
    df["Accuracy Label"] = df["Accuracy"].map({0.0: "Incorrect (Acc=0)", 1.0: "Correct (Acc=1)"})

    df_acc0 = df[df["Accuracy"] == 0.0]
    df_acc1 = df[df["Accuracy"] == 1.0]

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

    # --- Breakdown by Level ---
    print("\n" + "="*80)
    print("      SUMMARY BY DIFFICULTY LEVEL (ALL METRICS)")
    print("="*80)
    level_summary = df.groupby('Level', observed=True).agg(
        Total_Ans=('Accuracy', 'count'),
        Correct_Ans=('Accuracy', 'sum'),
        Acc_Pct=('Accuracy', lambda x: x.mean() * 100),
        Format_Pct=('Format', lambda x: x.mean() * 100),
        Avg_LC=('LC Score', 'mean'),
        Avg_BTs=('Backtracks', 'mean'),
        Avg_Len=('Length', 'mean')
    ).reset_index()
    format_mapping = {'Acc_Pct': '{:.1f}%', 'Format_Pct': '{:.1f}%', 'Avg_LC': '{:.2f}', 'Avg_BTs': '{:.2f}', 'Avg_Len': '{:.0f}'}
    for col, fmt in format_mapping.items():
        level_summary[col] = level_summary[col].map(fmt.format)
    print(level_summary.to_string(index=False))

    # --- Breakdown by Level AND Language ---
    print("\n" + "="*90)
    print("      SUMMARY BY LEVEL AND LANGUAGE (ALL METRICS)")
    print("="*90)
    level_lang_summary = df.groupby(['Level', 'Language'], observed=True).agg(
        Total_Ans=('Accuracy', 'count'),
        Acc_Pct=('Accuracy', lambda x: x.mean() * 100),
        Format_Pct=('Format', lambda x: x.mean() * 100),
        Avg_LC=('LC Score', 'mean'),
        Avg_BTs=('Backtracks', 'mean'),
        Avg_Len=('Length', 'mean')
    ).reset_index()
    for col, fmt in format_mapping.items():
        level_lang_summary[col] = level_lang_summary[col].map(fmt.format)
    print(level_lang_summary.to_string(index=False))

    # --- Breakdown by Level AND Accuracy ---
    print("\n" + "="*90)
    print("      SUMMARY BY LEVEL AND ACCURACY (ALL METRICS)")
    print("="*90)
    level_acc_summary = df.groupby(['Level', 'Accuracy Label'], observed=True).agg(
        Total_Ans=('Accuracy', 'count'),
        Format_Pct=('Format', lambda x: x.mean() * 100),
        Avg_LC=('LC Score', 'mean'),
        Avg_BTs=('Backtracks', 'mean'),
        Avg_Len=('Length', 'mean')
    ).reset_index()
    format_mapping_acc = {'Format_Pct': '{:.1f}%', 'Avg_LC': '{:.2f}', 'Avg_BTs': '{:.2f}', 'Avg_Len': '{:.0f}'}
    for col, fmt in format_mapping_acc.items():
        level_acc_summary[col] = level_acc_summary[col].map(fmt.format)
    print(level_acc_summary.to_string(index=False))

    # --- Visualizations (x = Level, ordered low -> top) ---
    os.makedirs("./Plots", exist_ok=True)
    palette = {"Incorrect (Acc=0)": "#ff9999", "Correct (Acc=1)": "#66b3ff"}
    strip_palette = {"Incorrect (Acc=0)": "#e63946", "Correct (Acc=1)": "#1d3557"}

    # 1. Format Compliance Plot (Bar Plot of Percentages)
    print("\nGenerating Bar Plot for Format Compliance Percentages...")
    plt.figure(figsize=(10, 7))
    df["Format Percentage"] = df["Format"] * 100
    sns.barplot(
        data=df, x="Level", y="Format Percentage", hue="Accuracy Label",
        palette=palette, edgecolor="black", alpha=0.9, capsize=0.1
    )
    plt.title("Percentage of Answers with Correct Formatting by Difficulty Level", fontsize=15, pad=15)
    plt.xlabel("Difficulty Level", fontsize=12)
    plt.ylabel("% of Answers with Correct Format", fontsize=12)
    plt.ylim(0, 105)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.legend(title="Accuracy Status", loc="lower right", framealpha=0.9)
    plt.savefig("./Plots/format_compliance_by_level_barplot.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 2. Language Consistency Boxplot
    print("Generating Boxplot/Stripplot for Language Consistency...")
    plt.figure(figsize=(10, 7))
    sns.boxplot(
        data=df, x="Level", y="LC Score", hue="Accuracy Label",
        palette=palette, showmeans=True,
        meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "black"},
        fliersize=0, boxprops={"alpha": 0.7}
    )
    sns.stripplot(
        data=df, x="Level", y="LC Score", hue="Accuracy Label",
        palette=strip_palette, dodge=True, jitter=0.25, alpha=0.4, linewidth=0, size=4, legend=False
    )
    plt.title("Language Consistency Scores by Difficulty Level and Accuracy", fontsize=15, pad=15)
    plt.xlabel("Difficulty Level", fontsize=12)
    plt.ylabel("Language Consistency Score (0.0 to 1.0)", fontsize=12)
    plt.ylim(-0.05, 1.05)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.legend(title="Accuracy Status", loc="lower right", framealpha=0.9)
    plt.savefig("./Plots/language_consistency_by_level_boxplot.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 3. Backtracks Plot
    print("Generating Boxplot/Stripplot for Backtracks...")
    plt.figure(figsize=(10, 7))
    sns.boxplot(
        data=df, x="Level", y="Backtracks", hue="Accuracy Label",
        palette=palette, showmeans=True,
        meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "black"},
        fliersize=0, boxprops={"alpha": 0.7}
    )
    sns.stripplot(
        data=df, x="Level", y="Backtracks", hue="Accuracy Label",
        palette=strip_palette, dodge=True, jitter=0.25, alpha=0.4, linewidth=0, size=4, legend=False
    )
    plt.title("Number of Backtracks by Difficulty Level and Accuracy", fontsize=15, pad=15)
    plt.xlabel("Difficulty Level", fontsize=12)
    plt.ylabel("Total Backtracks", fontsize=12)
    plt.ylim(-0.5, df["Backtracks"].max() + 1.5)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.legend(title="Accuracy Status", loc="upper right", framealpha=0.9)
    plt.savefig("./Plots/backtracks_by_level_boxplot.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 4. Token Length Plot
    print("Generating Boxplot/Stripplot for Answer Token Length...")
    plt.figure(figsize=(10, 7))
    sns.boxplot(
        data=df, x="Level", y="Length", hue="Accuracy Label",
        palette=palette, showmeans=True,
        meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "black"},
        fliersize=0, boxprops={"alpha": 0.7}
    )
    sns.stripplot(
        data=df, x="Level", y="Length", hue="Accuracy Label",
        palette=strip_palette, dodge=True, jitter=0.25, alpha=0.4, linewidth=0, size=3, legend=False
    )
    plt.title("Answer Length (Tokens) by Difficulty Level and Accuracy", fontsize=15, pad=15)
    plt.xlabel("Difficulty Level", fontsize=12)
    plt.ylabel("Length of Answer (Tokens)", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.legend(title="Accuracy Status", loc="upper right", framealpha=0.9)
    plt.savefig("./Plots/length_by_level_boxplot.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 5. Accuracy by Level and Language
    print("Generating Bar Plot for Accuracy by Level and Language...")
    plt.figure(figsize=(10, 7))
    acc_by_level_lang = df.groupby(['Level', 'Language'], observed=True)['Accuracy'].mean().reset_index()
    acc_by_level_lang['Accuracy'] *= 100
    sns.barplot(data=acc_by_level_lang, x="Level", y="Accuracy", hue="Language", palette="Set2", edgecolor="black")
    plt.title("Accuracy (%) by Difficulty Level and Language", fontsize=15, pad=15)
    plt.xlabel("Difficulty Level", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.ylim(0, 105)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.legend(title="Language")
    plt.savefig("./Plots/accuracy_by_level_barplot.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("\nPlots saved to ./Plots/")


if __name__ == "__main__":
    main()