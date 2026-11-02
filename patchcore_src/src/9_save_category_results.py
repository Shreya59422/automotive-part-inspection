
import pandas as pd
from pathlib import Path

results = {
    "Category": [
        "broken_large",
        "broken_small",
        "contamination"
    ],
    "Images": [20, 22, 21],
    "Precision": [0.6636, 0.5963, 0.7310],
    "Recall": [0.9369, 0.9206, 0.6804],
    "F1 Score": [0.7725, 0.7170, 0.5761],
    "IoU": [0.6317, 0.5639, 0.4417]
}

df = pd.DataFrame(results)

output_dir = Path("results")
output_dir.mkdir(exist_ok=True)

csv_path = output_dir / "category_evaluation_results.csv"

df.to_csv(csv_path, index=False)

print("Category evaluation results saved!")
print(f"File: {csv_path}")

print("\nResults:")
print(df.to_string(index=False))
