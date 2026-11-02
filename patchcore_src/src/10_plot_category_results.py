
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

csv_path = Path("results/category_evaluation_results.csv")

df = pd.read_csv(csv_path)

metrics = ["Precision", "Recall", "F1 Score", "IoU"]

for metric in metrics:
    plt.figure(figsize=(8, 5))

    plt.bar(df["Category"], df[metric])

    plt.title(f"{metric} by Defect Category")
    plt.xlabel("Defect Category")
    plt.ylabel(metric)

    plt.ylim(0, 1)
    plt.xticks(rotation=15)
    plt.tight_layout()

    output_path = Path("results") / f"{metric.lower().replace(' ', '_')}_comparison.png"

    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved: {output_path}")

print("\nAll comparison charts created!")
