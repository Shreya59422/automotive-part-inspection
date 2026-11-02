
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# Project paths
ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "results" / "patchcore_category_comparison.csv"
OUTPUT_DIR = ROOT / "results" / "patchcore_comparison"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load evaluation results
df = pd.read_csv(CSV_PATH)

if "Category" not in df.columns:
    raise ValueError(
        "The CSV must contain a 'Category' column."
    )

metrics = {
    "image_AUROC": "Image AUROC by Category",
    "image_F1Score": "Image F1-Score by Category",
    "pixel_AUROC": "Pixel AUROC by Category",
    "pixel_F1Score": "Pixel F1-Score by Category",
}

# Validate metric columns
for metric in metrics:
    if metric not in df.columns:
        raise ValueError(f"Missing metric column: {metric}")

# Create one graph per metric
for metric, title in metrics.items():
    data = df.sort_values(metric, ascending=False)

    plt.figure(figsize=(12, 6))
    bars = plt.bar(data["Category"], data[metric])

    plt.title(title)
    plt.xlabel("MVTec AD Category")
    plt.ylabel("Score")
    plt.ylim(0, 1.05)
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.4)

    # Display scores above bars
    for bar, score in zip(bars, data[metric]):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.015,
            f"{score:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()
    output_path = OUTPUT_DIR / f"{metric}.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")

print("\nAll four comparison graphs generated successfully.")
print(f"Categories evaluated: {df['Category'].nunique()}")
