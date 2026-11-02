
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Paths
DATA_DIR = Path("data/mvtec/bottle")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Image transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# Training dataset: good images only
train_dataset = datasets.ImageFolder(
    DATA_DIR / "train",
    transform=transform
)

# Test dataset: good and defective images
test_dataset = datasets.ImageFolder(
    DATA_DIR / "test",
    transform=transform
)

# Convert test labels:
# good = 0
# all defects = 1
test_dataset.targets = [
    0 if name == "good" else 1
    for name in test_dataset.classes
    for _ in range(len(list((DATA_DIR / "test" / name).glob("*"))))
]

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False
)

# CNN model
class DefectCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


model = DefectCNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

# Training
epochs = 10

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        images = images.to(device)

        # Training images are all good.
        labels = torch.zeros(
            images.size(0),
            dtype=torch.long,
            device=device
        )

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(
        f"Epoch [{epoch + 1}/{epochs}], "
        f"Loss: {total_loss / len(train_loader):.4f}"
    )

# Save model
torch.save(
    model.state_dict(),
    MODEL_DIR / "bottle_binary_cnn.pth"
)

print("Training completed.")
print("Model saved to models/bottle_binary_cnn.pth")
