
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Load threshold results
file_path = Path("results/threshold_analysis.csv")
df = pd.read_csv(file_path)

# Create output folder
output_folder = Path("results")
output_folder.mkdir(exist_ok=True)

# Metrics to plot
metrics = ["Precision", "Recall", "F1 Score", "FNR"]

for metric in metrics:
    plt.figure(figsize=(8, 5))

    plt.plot(
        df["Threshold"],
        df[metric],
        marker="o",
        linewidth=2,
    )

    plt.title(f"{metric} vs Anomaly Threshold")
    plt.xlabel("Anomaly Threshold")
    plt.ylabel(metric)
    plt.xticks(df["Threshold"])
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle="--", alpha=0.5)

    output_path = output_folder / f"threshold_{metric.lower().replace(' ', '_')}.png"

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")

print("\nAll threshold comparison graphs created!")
