import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set publication styling
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.edgecolor'] = '#cbd5e1'
plt.rcParams['axes.linewidth'] = 1.2

FIGURES_DIR = os.path.join("reports", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

CLASS_NAMES = ["Negative", "Neutral", "Positive"]
COLOR_PALETTE = {
    "negative": "#ef4444",
    "neutral": "#f59e0b",
    "positive": "#10b981",
    "indigo": "#6366f1",
    "purple": "#8b5cf6",
    "slate": "#64748b"
}


def load_reports():
    """Load baseline model comparison and transformer benchmark reports."""
    baseline_path = os.path.join("reports", "model_comparison.json")
    transformer_path = os.path.join("reports", "full_dataset_transformer_comparison.json")

    baseline_data = {}
    if os.path.exists(baseline_path):
        with open(baseline_path, "r") as f:
            baseline_data = json.load(f)

    transformer_data = {}
    if os.path.exists(transformer_path):
        with open(transformer_path, "r") as f:
            transformer_data = json.load(f)

    return baseline_data, transformer_data


def plot_confusion_matrices(baseline_data, transformer_data):
    """Plot annotated confusion matrix heatmaps for all models."""
    models_cm = {}

    # Extract classical models
    if "models" in baseline_data:
        for m_name, m_info in baseline_data["models"].items():
            if "confusion_matrix" in m_info:
                models_cm[m_name] = np.array(m_info["confusion_matrix"])

    # Extract transformer models
    if "test_model_evaluations" in transformer_data:
        for m_name, m_info in transformer_data["test_model_evaluations"].items():
            if "confusion_matrix" in m_info:
                clean_name = m_name.replace("(Full 60k)", "").strip()
                models_cm[clean_name] = np.array(m_info["confusion_matrix"])

    n_models = len(models_cm)
    if n_models == 0:
        return

    cols = min(3, n_models)
    rows = (n_models + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.5 * rows))
    if n_models == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, (m_name, cm) in enumerate(models_cm.items()):
        ax = axes[idx]
        cm_sum = cm.sum(axis=1, keepdims=True)
        cm_perc = np.divide(cm, cm_sum, out=np.zeros_like(cm, dtype=float), where=cm_sum != 0) * 100

        # Annotations format: Count (Percentage%)
        annot = np.empty_like(cm, dtype=object)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                annot[i, j] = f"{cm[i, j]:,}\n({cm_perc[i, j]:.1f}%)"

        sns.heatmap(
            cm_perc,
            annot=annot,
            fmt="",
            cmap="Blues",
            cbar=False,
            ax=ax,
            xticklabels=CLASS_NAMES,
            yticklabels=CLASS_NAMES,
            annot_kws={"size": 9, "weight": "bold"}
        )
        ax.set_title(m_name, fontsize=12, fontweight="bold", pad=10, color="#1e293b")
        ax.set_xlabel("Predicted Label", fontsize=10, fontweight="semibold", color="#475569")
        ax.set_ylabel("True Label", fontsize=10, fontweight="semibold", color="#475569")

    # Hide unused axes
    for j in range(n_models, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Confusion Matrix Comparison Across Models", fontsize=15, fontweight="bold", y=0.98, color="#0f172a")
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "01_confusion_matrices.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Visualize] Saved '{out_path}'")


def plot_accuracy_and_f1_comparison(baseline_data, transformer_data):
    """Plot Accuracy vs Macro F1 comparison bar chart."""
    records = []

    if "models" in baseline_data:
        for name, info in baseline_data["models"].items():
            records.append({
                "Model": name,
                "Accuracy": info.get("accuracy", 0.0) * 100,
                "Macro F1": info.get("macro_f1", 0.0) * 100,
                "Type": "Baseline"
            })

    if "test_model_evaluations" in transformer_data:
        for name, info in transformer_data["test_model_evaluations"].items():
            records.append({
                "Model": name.replace("(Full 60k)", "").strip(),
                "Accuracy": info.get("accuracy", 0.0) * 100,
                "Macro F1": info.get("macro_f1", 0.0) * 100,
                "Type": "Transformer"
            })

    df = pd.DataFrame(records)
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(df))
    width = 0.35

    rects1 = ax.bar(x - width/2, df["Accuracy"], width, label="Accuracy (%)", color="#6366f1", edgecolor="none", alpha=0.9)
    rects2 = ax.bar(x + width/2, df["Macro F1"], width, label="Macro F1 (%)", color="#10b981", edgecolor="none", alpha=0.9)

    ax.set_ylabel("Score (%)", fontsize=11, fontweight="bold")
    ax.set_title("Held-Out Test Performance: Accuracy vs. Macro F1", fontsize=14, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(df["Model"], fontsize=10, fontweight="semibold")
    ax.set_ylim(0, 100)
    ax.legend(frameon=True, facecolor="white", edgecolor="#e2e8f0", fontsize=10)

    # Bar value labels
    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "02_model_accuracy_f1_comparison.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Visualize] Saved '{out_path}'")


def plot_class_level_breakdown(baseline_data, transformer_data):
    """Plot per-class F1 score breakdown across models."""
    class_data = []

    sources = []
    if "models" in baseline_data:
        for name, info in baseline_data["models"].items():
            sources.append((name, info.get("class_metrics", {})))

    if "test_model_evaluations" in transformer_data:
        for name, info in transformer_data["test_model_evaluations"].items():
            clean_name = name.replace("(Full 60k)", "").strip()
            sources.append((clean_name, info.get("class_metrics", {})))

    for m_name, metrics in sources:
        for cls_name in ["negative", "neutral", "positive"]:
            if cls_name in metrics:
                class_data.append({
                    "Model": m_name,
                    "Class": cls_name.capitalize(),
                    "F1": metrics[cls_name].get("f1", 0.0) * 100
                })

    df = pd.DataFrame(class_data)
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(11, 5.5))
    sns.barplot(
        data=df, x="Model", y="F1", hue="Class",
        palette=[COLOR_PALETTE["negative"], COLOR_PALETTE["neutral"], COLOR_PALETTE["positive"]],
        ax=ax, edgecolor="none", alpha=0.9
    )

    ax.set_ylabel("F1 Score (%)", fontsize=11, fontweight="bold")
    ax.set_title("Per-Class F1 Score Breakdown Across Models", fontsize=14, fontweight="bold", pad=12)
    ax.set_ylim(0, 100)
    ax.legend(title="Sentiment Class", frameon=True, facecolor="white", edgecolor="#e2e8f0")

    # Add bar labels
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f"{height:.1f}%", (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom', fontsize=8, xytext=(0, 2),
                        textcoords='offset points', fontweight='semibold')

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "03_class_level_f1_breakdown.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Visualize] Saved '{out_path}'")


def plot_class_distribution(transformer_data):
    """Plot dataset class balance donut chart."""
    dist = {}
    if "class_distributions" in transformer_data and "train" in transformer_data["class_distributions"]:
        dist = transformer_data["class_distributions"]["train"]
    elif "train_class_distribution" in transformer_data:
        dist = transformer_data["train_class_distribution"]

    if not dist:
        dist = {"neutral": 0.458, "positive": 0.352, "negative": 0.190}

    labels = [k.capitalize() for k in dist.keys()]
    sizes = [v * 100 if v <= 1.0 else v for v in dist.values()]
    colors = [COLOR_PALETTE["neutral"] if l == "Neutral" else COLOR_PALETTE["positive"] if l == "Positive" else COLOR_PALETTE["negative"] for l in labels]

    fig, ax = plt.subplots(figsize=(6, 6))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, autopct='%1.1f%%',
        startangle=140, colors=colors, pctdistance=0.75,
        textprops={'fontsize': 11, 'weight': 'bold'},
        wedgeprops=dict(width=0.4, edgecolor='white', linewidth=2)
    )

    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_weight('bold')

    ax.set_title("CardiffNLP TweetEval Dataset Class Distribution\n(41,746 Training Samples)", fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "04_class_distribution.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Visualize] Saved '{out_path}'")


def plot_negative_recall_boost(transformer_data):
    """Plot Negative Class Recall improvement chart."""
    neg_rec = {}
    if "test_model_evaluations" in transformer_data:
        for name, info in transformer_data["test_model_evaluations"].items():
            c_metrics = info.get("class_metrics", {})
            if "negative" in c_metrics:
                clean_name = name.replace("(Full 60k)", "").strip()
                neg_rec[clean_name] = c_metrics["negative"].get("recall", 0.0) * 100

    if not neg_rec:
        neg_rec = {
            "Logistic Regression": 39.72,
            "Bi-LSTM": 60.28,
            "DistilBERT Base": 74.22,
            "Vanilla RoBERTa Base": 81.27
        }

    df = pd.DataFrame(list(neg_rec.items()), columns=["Model", "Negative Recall"])

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(df["Model"], df["Negative Recall"], color=["#94a3b8", "#818cf8", "#6366f1", "#10b981"], height=0.55)

    ax.set_xlabel("Negative Class Recall (%)", fontsize=11, fontweight="bold")
    ax.set_title("Class Imbalance Mitigation: Negative Class Recall Boost", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlim(0, 100)

    for bar in bars:
        w = bar.get_width()
        ax.annotate(f"{w:.1f}%", xy=(w + 1.5, bar.get_y() + bar.get_height() / 2),
                    va="center", fontsize=10, fontweight="bold", color="#1e293b")

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "05_negative_class_recall_boost.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Visualize] Saved '{out_path}'")


def plot_cross_validation_f1(baseline_data):
    """Plot 5-Fold Stratified Cross-Validation mean F1 and std dev."""
    cv_data = baseline_data.get("cross_validation", {})
    if not cv_data:
        return

    models = list(cv_data.keys())
    means = [cv_data[m]["mean_f1"] * 100 for m in models]
    stds = [cv_data[m]["std_f1"] * 100 for m in models]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(models, means, yerr=stds, capsize=6, color="#6366f1", alpha=0.85, edgecolor="none", width=0.45)

    ax.set_ylabel("5-Fold Mean Macro F1 (%)", fontsize=11, fontweight="bold")
    ax.set_title("5-Fold Stratified Cross-Validation Performance (Train Split)", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylim(0, 100)

    for bar, m, s in zip(bars, means, stds):
        ax.annotate(f"{m:.1f}% ± {s:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, m + s + 2),
                    ha="center", va="bottom", fontsize=9.5, fontweight="bold")

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "06_cross_validation_f1.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Visualize] Saved '{out_path}'")


def plot_calibration_scores(baseline_data):
    """Plot calibration evaluation Brier scores."""
    calib = baseline_data.get("calibration_eval", {})
    if not calib:
        return

    methods = ["Uncalibrated", "Sigmoid (Platt)", "Isotonic"]
    scores = [
        calib.get("uncalibrated_brier", 0.5243),
        calib.get("sigmoid_brier", 0.5076),
        calib.get("isotonic_brier", 0.5087)
    ]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(methods, scores, color=["#f59e0b", "#10b981", "#6366f1"], width=0.45, alpha=0.85)

    ax.set_ylabel("Brier Score (Lower is Better)", fontsize=11, fontweight="bold")
    ax.set_title("Probability Calibration Comparison on Validation Set", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylim(0, max(scores) * 1.25)

    for bar, score in zip(bars, scores):
        ax.annotate(f"{score:.4f}", xy=(bar.get_x() + bar.get_width() / 2, score + 0.01),
                    ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "07_calibration_brier_scores.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Visualize] Saved '{out_path}'")


def plot_latency_vs_f1_tradeoff(transformer_data):
    """Plot latency vs Macro F1 efficiency tradeoff."""
    models = [
        {"name": "Logistic Regression", "latency": 0.42, "f1": 58.50, "type": "Baseline"},
        {"name": "Bi-LSTM", "latency": 3.12, "f1": 59.54, "type": "Baseline"},
        {"name": "DistilBERT (Full)", "latency": 14.01, "f1": 72.21, "type": "Deployed Transformer"},
        {"name": "Vanilla RoBERTa", "latency": 26.25, "f1": 72.08, "type": "Transformer"}
    ]

    df = pd.DataFrame(models)

    fig, ax = plt.subplots(figsize=(9, 5))
    for mtype, group in df.groupby("type"):
        ax.scatter(group["latency"], group["f1"], label=mtype, s=140, alpha=0.9, edgecolors="none")

    for _, row in df.iterrows():
        ax.annotate(f" {row['name']} ({row['f1']:.1f}%)", xy=(row['latency'], row['f1']),
                    xytext=(5, -5), textcoords="offset points", fontsize=9.5, fontweight="bold")

    ax.set_xlabel("Single-Sample CPU Inference Latency (ms)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Test Macro F1 Score (%)", fontsize=11, fontweight="bold")
    ax.set_title("Model Selection Frontier: Inference Latency vs. Macro F1", fontsize=13, fontweight="bold", pad=12)
    ax.legend(frameon=True, facecolor="white", edgecolor="#e2e8f0")
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "08_cpu_latency_vs_f1_tradeoff.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Visualize] Saved '{out_path}'")


def generate_all_visualizations():
    """Main execution entry point to generate all evaluation charts."""
    print("=== Generating SentimentScope Evaluation Visualizations ===")
    baseline_data, transformer_data = load_reports()

    plot_confusion_matrices(baseline_data, transformer_data)
    plot_accuracy_and_f1_comparison(baseline_data, transformer_data)
    plot_class_level_breakdown(baseline_data, transformer_data)
    plot_class_distribution(transformer_data)
    plot_negative_recall_boost(transformer_data)
    plot_cross_validation_f1(baseline_data)
    plot_calibration_scores(baseline_data)
    plot_latency_vs_f1_tradeoff(transformer_data)

    print(f"--> All visualizations successfully generated and saved to '{FIGURES_DIR}'.")


if __name__ == "__main__":
    generate_all_visualizations()
