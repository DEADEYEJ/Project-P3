import collections
import math
import os
import shutil
import pandas as pd
import torch
import torchvision
from torch import nn
from d2l import torch as d2l
from pathlib import Path


# Tilpas disse stier, når vi kender din mappestruktur
IMAGE_DIR = Path("sti/til/billeder")
CSV_FILE = Path("sti/til/labels.csv")

# CSV'en skal på dette trin have kolonnerne:
# filename, labels, video_id
# labels kan fx være "erosion;crack"
df = pd.read_csv(CSV_FILE)

# Find alle kategorier i datasættet
classes = sorted({
    label.strip()
    for labels in df["labels"].dropna()
    for label in labels.split(";")
})
class_to_idx = {name: i for i, name in enumerate(classes)}

print("Kategorier:", classes)

"""Del derefter data efter vidoe, så frames fra samme video ikke 
blandes mellem træning, validering og test."""

# Først: hold 20 % af videoerne til test
split1 = GroupShuffleSplit(
    n_splits=1, test_size=0.2, random_state=42
)
train_val_idx, test_idx = next(
    split1.split(df, groups=df["video_id"])
)

train_val_df = df.iloc[train_val_idx].reset_index(drop=True)
test_df = df.iloc[test_idx].reset_index(drop=True)

# Dernæst: hold ca. 20 % af de resterende videoer til validering
split2 = GroupShuffleSplit(
    n_splits=1, test_size=0.25, random_state=42
)
train_idx, val_idx = next(
    split2.split(train_val_df, groups=train_val_df["video_id"])
)

train_df = train_val_df.iloc[train_idx].reset_index(drop=True)
val_df = train_val_df.iloc[val_idx].reset_index(drop=True)

print("Træningsbilleder:", len(train_df))
print("Valideringsbilleder:", len(val_df))
print("Testbilleder:", len(test_df))

"""Lav billedetransformationer. Augmentation bruges kun til træning, ikke til validering eller test."""

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

"""Datasætkoden læser billederne og laver en target-vektor med en
plads per kategori. Et 1 betyder, at kategorien findes i billedet."""

class BladeDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform, class_to_idx):
        self.dataframe = dataframe
        self.image_dir = image_dir
        self.transform = transform
        self.class_to_idx = class_to_idx

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]

        image = Image.open(self.image_dir / row["filename"]).convert("RGB")
        image = self.transform(image)

        target = torch.zeros(len(self.class_to_idx))
        labels = str(row["labels"]).split(";")

        for label in labels:
            label = label.strip()
            if label in self.class_to_idx:
                target[self.class_to_idx[label]] = 1.0

        return image, target


train_dataset = BladeDataset(
    train_df, IMAGE_DIR, train_transform, class_to_idx
)
val_dataset = BladeDataset(
    val_df, IMAGE_DIR, eval_transform, class_to_idx
)
test_dataset = BladeDataset(
    test_df, IMAGE_DIR, eval_transform, class_to_idx
)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)


"""Her er en ResNet-18-model, der er fortrænet på ImageNet og tilpasset
jeres kategorier. BCEWithLogitsLoss passer til multi-label klassifikation."""


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# Udskift modellens sidste lag, så det passer til jeres antal kategorier
model.fc = nn.Linear(model.fc.in_features, len(classes))
model = model.to(device)

loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

"""Træn modellen, og mål valideringstabet efter hver runde"""

epochs = 10

for epoch in range(epochs):
    model.train()
    train_loss = 0.0

    for images, targets in train_loader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        predictions = model(images)
        loss = loss_fn(predictions, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            targets = targets.to(device)

            predictions = model(images)
            val_loss += loss_fn(predictions, targets).item()

    print(
        f"Epoch {epoch + 1}/{epochs} "
        f"train loss: {train_loss / len(train_loader):.4f} "
        f"val loss: {val_loss / len(val_loader):.4f}"
    )

    """Når modellen er trænet, kan den forudsige alle kategorier,
    den mener findes i et billede"""

    model.eval()

images, targets = next(iter(test_loader))
images = images.to(device)

with torch.no_grad():
    probabilities = torch.sigmoid(model(images))

threshold = 0.5

for probs in probabilities.cpu():
    predicted_classes = [
        classes[i]
        for i, probability in enumerate(probs)
        if probability >= threshold
    ]
    print(predicted_classes)