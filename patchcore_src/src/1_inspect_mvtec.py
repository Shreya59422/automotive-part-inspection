
from pathlib import Path
from PIL import Image

# Dataset path
DATASET_PATH = Path("data/mvtec/bottle")

# Test categories
categories = ["good", "broken_large", "broken_small", "contamination"]

print("MVTec Bottle Dataset Inspection")
print("=" * 40)

# Check training images
train_path = DATASET_PATH / "train" / "good"
train_images = list(train_path.glob("*"))

print(f"\nTraining good images: {len(train_images)}")

# Check test images
print("\nTest image counts:")

for category in categories:
    folder = DATASET_PATH / "test" / category
    images = list(folder.glob("*"))

    print(f"{category}: {len(images)}")

# Inspect image dimensions
print("\nImage dimensions:")

if train_images:
    image_path = train_images[0]

    with Image.open(image_path) as image:
        print("Sample image:", image_path.name)
        print("Image size:", image.size)
        print("Image mode:", image.mode)

print("\nDataset inspection completed.")
