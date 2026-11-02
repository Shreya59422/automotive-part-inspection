from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

results_dir = Path("results")
csv_path = results_dir / "patchcore_all_categories.csv"

if not csv_path.exists():
    raise FileNotFoundError(f"Results file not found: {csv_path}")

df = pd.read_csv(csv_path)

metric_columns = [
    "image_AUROC",
    "image_F1Score",
    "pixel_AUROC",
    "pixel_F1Score",
]

required_columns = ["Category", *metric_columns]
missing = set(required_columns) - set(df.columns)

if missing:
    raise ValueError(f"Missing CSV columns: {sorted(missing)}")

for col in metric_columns:
    df[col] = pd.to_numeric(df[col], errors="raise")

# Convert metric values to percentages for reporting.
report = df[["Category", *metric_columns]].copy()

print("\nCATEGORY-WISE RESULTS")
print(
    report.to_string(
        index=False,
        formatters={
            col: lambda x: f"{x * 100:.2f}%"
            for col in metric_columns
        },
    )
)

print("\nMACRO-AVERAGE ACROSS CATEGORIES")
for col in metric_columns:
    print(f"{col}: {df[col].mean() * 100:.2f}%")

# Rank categories from weakest to strongest localization.
ranked = report.sort_values("pixel_F1Score")
ranked.to_csv(
    results_dir / "patchcore_category_metrics_ranked.csv",
    index=False,
)

# Generate a localization performance chart.
plot_df = ranked.sort_values("pixel_F1Score")

plt.figure(figsize=(10, 7))
plt.barh(
    plot_df["Category"],
    plot_df["pixel_F1Score"] * 100,
)
plt.xlabel("Pixel F1-score (%)")
plt.ylabel("MVTec AD category")
plt.title("PatchCore: Pixel-Level Localization Performance")
plt.tight_layout()
plt.savefig(
    results_dir / "patchcore_pixel_f1_by_category.png",
    dpi=200,
)
plt.close()

print("\nSaved:")
print("results/patchcore_category_metrics_ranked.csv")
print("results/patchcore_pixel_f1_by_category.png")