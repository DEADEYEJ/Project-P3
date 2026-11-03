import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import wandb
from torch.utils.data import DataLoader

wandb.login()

batch_size = 16
learningRate = 0.001
Momentum = 0.9
epochrange = 50

project = "P3-Project"

config = {
    "epochs": epochrange,
    "lr": learningRate,
    "batch_size": batch_size,
    "momentum": Momentum,
    "optimizer": "SGD",
    "architecture": "CNN"
}

# None RandomSeed
seed = 42

random.seed(seed)
torch.manual_seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# Device
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {device}")


transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])

trainset = torchvision.datasets.ImageFolder(
    root="Dataset/train",
    transform=transform
)

validationset = torchvision.datasets.ImageFolder(
    root="Dataset/validation",
    transform=transform
)

num_classes = len(trainset.classes)

print(f"Number of classes: {num_classes}")
print(f"Training images: {len(trainset)}")
print(f"validation images: {len(validationset)}")

trainloader = DataLoader(
    trainset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
    pin_memory=(device.type == "cuda")
)

validationloader = DataLoader(
    validationset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
    pin_memory=(device.type == "cuda")
)


class Net(nn.Module):

    def __init__(self, num_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(6, 16, 5)

        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        x = self.fc3(x)

        return x


net = Net(num_classes).to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(
    net.parameters(),
    lr=learningRate,
    momentum=Momentum
)

def evaluate(net, validationloader, criterion):

    net.eval()

    validation_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for inputs, labels in validationloader:

            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = net(inputs)

            loss = criterion(outputs, labels)

            validation_loss += loss.item() * labels.size(0)

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)

            correct += (predicted == labels).sum().item()

    average_loss = validation_loss / total
    accuracy = 100 * correct / total

    net.train()

    return average_loss, accuracy

wandb.init(
    project=project,
    config=config
)

wandb.watch(
    net,
    log="all",
    log_freq=10
)

def main():

    global_step = 0
    best_accuracy = 0.0

    for epoch in range(epochrange):

        net.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in trainloader:

            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = net(inputs)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            _, predicted = torch.max(outputs, 1)

            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            running_loss += loss.item() * labels.size(0)

            # Batch metrics
            wandb.log({
                "train/batch_loss": loss.item(),
                "epoch": epoch
            }, step=global_step)

            global_step += 1

        train_loss = running_loss / total

        train_accuracy = 100 * correct / total

        validation_loss, validation_accuracy = evaluate(
            net,
            validationloader,
            criterion
        )

        wandb.log({

            "train/epoch_loss": train_loss,
            "train/epoch_accuracy": train_accuracy,

            "validation/loss": validation_loss,
            "validation/accuracy": validation_accuracy,

            "epoch": epoch

        }, step=global_step)

        print(
            f"Epoch [{epoch + 1}/{epochrange}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.2f}% | "
            f"validation Loss: {validation_loss:.4f} | "
            f"validation Acc: {validation_accuracy:.2f}%"
        )
 
        if validation_accuracy > best_accuracy:

            best_accuracy = validation_accuracy

            #model checkpoint
            checkpoint_path = "best_model.pth"

            torch.save(net.state_dict(), checkpoint_path)

            # Wandb Artifact / Checkpoint
            artifact = wandb.Artifact(
                name="best-model",
                type="model",
                description="Best CNN model checkpoint",
                metadata={
                    "epoch": epoch + 1,
                    "validation_accuracy": validation_accuracy,
                    "learning_rate": learningRate,
                    "batch_size": batch_size,
                    "classes": trainset.classes
                }
            )
            artifact.add_file(checkpoint_path)

            # Upload Wandb
            wandb.log_artifact(artifact)

            print(
                f"New best model uploaded"
                f"Accuracy: {best_accuracy:.2f}%"
            )


    print("Finished Training")
    print(f"Best validation accuracy: {best_accuracy:.2f}%")

    wandb.finish()


if __name__ == "__main__":
    main()
