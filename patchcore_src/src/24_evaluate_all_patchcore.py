
from pathlib import Path
import pandas as pd

from anomalib.data import MVTecAD
from anomalib.models import Patchcore
from anomalib.engine import Engine


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_ROOT = PROJECT_ROOT / "data" / "mvtec"

CHECKPOINT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "anomaly_detection"
    / "Patchcore"
    / "MVTecAD"
)

OUTPUT_CSV = PROJECT_ROOT / "results" / "patchcore_category_comparison.csv"

METRIC_NAMES = [
    "image_AUROC",
    "image_F1Score",
    "pixel_AUROC",
    "pixel_F1Score",
]

CATEGORIES = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
]

CSV_COLUMNS = ["Category"] + METRIC_NAMES


# --------------------------------------------------
# 2. Load existing results without losing them
# --------------------------------------------------

def load_existing_results():
    if not OUTPUT_CSV.exists():
        return {}

    existing_df = pd.read_csv(OUTPUT_CSV)
    results = {}

    for _, row in existing_df.iterrows():
        category = str(row["Category"]).strip().lower()

        if category not in CATEGORIES:
            continue

        results[category] = {
            "Category": str(row["Category"]),
            **{
                metric: row.get(metric, None)
                for metric in METRIC_NAMES
            },
        }

    return results


# --------------------------------------------------
# 3. Save results after each successful evaluation
# --------------------------------------------------

def save_results(results):
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    for category in CATEGORIES:
        if category in results:
            row = results[category].copy()
            row["Category"] = category.title()
            rows.append(row)

    df = pd.DataFrame(rows, columns=CSV_COLUMNS)
    df.to_csv(OUTPUT_CSV, index=False)


# --------------------------------------------------
# 4. Check whether metrics are already complete
# --------------------------------------------------

def has_complete_metrics(row):
    if row is None:
        return False

    for metric in METRIC_NAMES:
        value = row.get(metric)

        if value is None or pd.isna(value):
            return False

        try:
            float(value)
        except (TypeError, ValueError):
            return False

    return True


# --------------------------------------------------
# 5. Find an existing checkpoint
# --------------------------------------------------

def find_checkpoint(category):
    category_dir = CHECKPOINT_ROOT / category

    if not category_dir.exists():
        return None

    checkpoints = list(
        category_dir.glob("*/weights/lightning/model.ckpt")
    )

    if not checkpoints:
        return None

    # Prefer v0; otherwise use the first available version.
    checkpoints.sort(
        key=lambda path: (
            path.parts[-4].lower() != "v0",
            path.parts[-4].lower(),
        )
    )

    return checkpoints[0]


# --------------------------------------------------
# 6. Evaluate one category
# --------------------------------------------------

def evaluate_category(category, checkpoint):
    print(f"\n{'=' * 65}")
    print(f"Evaluating category: {category}")
    print(f"Checkpoint: {checkpoint}")
    print(f"{'=' * 65}")

    datamodule = MVTecAD(
        root=DATA_ROOT,
        category=category,
        train_batch_size=8,
        eval_batch_size=8,
        num_workers=0,
    )

    model = Patchcore()

    engine = Engine(
        default_root_dir=str(
            PROJECT_ROOT / "results" / "anomaly_detection"
        )
    )

    test_results = engine.test(
        model=model,
        datamodule=datamodule,
        ckpt_path=str(checkpoint),
    )

    if not test_results:
        raise RuntimeError("Anomalib returned no test results.")

    metrics = test_results[0]
    row = {"Category": category.title()}

    for metric in METRIC_NAMES:
        value = metrics.get(metric)

        if value is None:
            row[metric] = None
        else:
            row[metric] = float(value)

    if not has_complete_metrics(row):
        raise RuntimeError(
            f"One or more expected metrics are missing: {row}"
        )

    return row


# --------------------------------------------------
# 7. Main evaluation loop
# --------------------------------------------------

def main():
    if not DATA_ROOT.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {DATA_ROOT}"
        )

    results = load_existing_results()

    print("PatchCore evaluation for MVTec AD")
    print(f"Dataset: {DATA_ROOT}")
    print(f"Checkpoint directory: {CHECKPOINT_ROOT}")
    print(f"Results CSV: {OUTPUT_CSV}")
    print()

    completed = [
        category
        for category in CATEGORIES
        if has_complete_metrics(results.get(category))
    ]

    print(f"Categories already evaluated: {len(completed)}")
    if completed:
        print(", ".join(category.title() for category in completed))

    for category in CATEGORIES:

        # Do not repeat a completed evaluation.
        if has_complete_metrics(results.get(category)):
            print(f"\nSkipping {category}: all metrics are already saved.")
            continue

        checkpoint = find_checkpoint(category)

        if checkpoint is None:
            print(f"\nSkipping {category}: no checkpoint found.")
            continue

        try:
            row = evaluate_category(category, checkpoint)

            results[category] = row
            save_results(results)

            print(f"\nSUCCESS: {category.title()}")
            for metric in METRIC_NAMES:
                print(f"  {metric}: {row[metric]}")

            print(f"Saved updated results to: {OUTPUT_CSV}")

        except KeyboardInterrupt:
            print("\nEvaluation interrupted by user.")
            print("Previously saved results have been preserved.")
            save_results(results)
            break

        except Exception as error:
            print(f"\nFAILED: {category.title()}")
            print(f"Reason: {error}")
            print("Continuing to the next category.")
            continue

    print("\n" + "=" * 65)
    print("EVALUATION SUMMARY")
    print("=" * 65)

    save_results(results)

    if OUTPUT_CSV.exists():
        final_df = pd.read_csv(OUTPUT_CSV)
        print(final_df.to_string(index=False))
        print(f"\nCategories with saved results: {len(final_df)}")
        print(f"CSV location: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
