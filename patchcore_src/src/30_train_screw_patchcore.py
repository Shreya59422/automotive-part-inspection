
from pathlib import Path
import csv
import traceback

from anomalib.data import MVTecAD
from anomalib.models import Patchcore
from anomalib.engine import Engine


# CONFIGURATION

DATASET_PATH = Path("data/mvtec")

RESULTS_PATH = Path("results/screw_synthetic_experiment")

CSV_PATH = Path("results/patchcore_screw_synthetic_validation.csv")

CATEGORIES = ["screw"]


# CREATE RESULTS DIRECTORY

CSV_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)


# RESULTS STORAGE

results = []

failed_categories = []


# TRAIN AND TEST ALL CATEGORIES\

for index, category in enumerate(CATEGORIES, start=1):

    print("\n" + "=" * 60)
    print(f"CATEGORY {index}/{len(CATEGORIES)}: {category.upper()}")
    print("=" * 60)

    try:
        # 1. Dataset

        datamodule = MVTecAD(
    root=Path("data/mvtec"),
    category=category,
    train_batch_size=8,
    eval_batch_size=8,
    num_workers=0,
    val_split_mode="synthetic",
    val_split_ratio=0.2,
    seed=42,
)


        # 2. PatchCore model

        model = Patchcore()


        # 3. Engine

        engine = Engine(
            default_root_dir=RESULTS_PATH,
        )


        # 4. Train

        print(f"Training PatchCore for {category}...")

        engine.fit(
            model=model,
            datamodule=datamodule,
        )


        # 5. Test

        print(f"Testing PatchCore for {category}...")

        test_results = engine.test(
            model=model,
            datamodule=datamodule,
        )


        # 6. Extract metrics

        metrics = {}

        if test_results:

            for result in test_results:

                if isinstance(result, dict):

                    metrics.update(result)


        row = {
            "Category": category,
            "image_AUROC": metrics.get("image_AUROC"),
            "image_F1Score": metrics.get("image_F1Score"),
            "pixel_AUROC": metrics.get("pixel_AUROC"),
            "pixel_F1Score": metrics.get("pixel_F1Score"),
        }


        # 7. Store results

        results.append(row)

        print("\nCategory metrics:")

        for key, value in row.items():

            print(f"{key}: {value}")


        # 8. Save CSV after every category

        with CSV_PATH.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "Category",
                    "image_AUROC",
                    "image_F1Score",
                    "pixel_AUROC",
                    "pixel_F1Score",
                ],
            )

            writer.writeheader()

            writer.writerows(results)


        print(f"\nSaved results to: {CSV_PATH}")


    except Exception as error:

        print(f"\nERROR while processing {category}")

        print(error)

        traceback.print_exc()

        failed_categories.append(category)

        continue


# FINAL SUMMARY

print("\n" + "=" * 60)
print("ALL CATEGORY TRAINING COMPLETED")
print("=" * 60)

print(f"\nSuccessful categories: {len(results)}")

print(f"Failed categories: {len(failed_categories)}")

if failed_categories:

    print("\nCategories that failed:")

    for category in failed_categories:

        print("-", category)


print(f"\nFinal results CSV: {CSV_PATH}")

print("\nFinal metrics:")

for row in results:

    print(row)

