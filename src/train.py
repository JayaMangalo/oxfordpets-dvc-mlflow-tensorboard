# src/train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml
import json
import mlflow
import mlflow.pytorch

# Load hyperparameters from params.yaml
with open("src/params.yaml") as f:
    params = yaml.safe_load(f)

EPOCHS = params["epochs"]
IMG_SIZE = params["img_size"]
LEARNING_RATES = params["lr_list"]
BATCH_SIZES = params["batch_size_list"]

# Transforms
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# Dataset and DataLoader
data_dir = "data/oxford-iiit-pet/images"
dataset = datasets.ImageFolder(data_dir, transform=transform)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define model builders
def build_simple_cnn(num_classes):
    return nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(32 * (IMG_SIZE // 4) * (IMG_SIZE // 4), 128),
        nn.ReLU(),
        nn.Linear(128, num_classes)
    )

def build_mobilenet(num_classes):
    model = models.mobilenet_v2(pretrained=True)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    return model

def build_resnet18(num_classes):
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

model_builders = {
    "simple_cnn": build_simple_cnn,
    "mobilenet_v2": build_mobilenet,
    "resnet18": build_resnet18
}

mlflow.set_experiment("OxfordPets-MultiModel-Tracking")

for model_name, builder in model_builders.items():
    for lr in LEARNING_RATES:
        for batch_size in BATCH_SIZES:
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            model = builder(len(dataset.classes)).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=lr)

            
            
            with mlflow.start_run(run_name=f"{model_name}-lr{lr}-bs{batch_size}"):
                mlflow.log_param("model", model_name)
                mlflow.log_param("epochs", EPOCHS)
                mlflow.log_param("img_size", IMG_SIZE)
                mlflow.log_param("lr", lr)
                mlflow.log_param("batch_size", batch_size)
                writer = SummaryWriter(log_dir=f"logs/{model_name}_lr{lr}_bs{batch_size}")
                
                for epoch in range(EPOCHS):
                    model.train()
                    running_loss = 0.0
                    correct = 0
                    total = 0

                    for images, labels in dataloader:
                        images, labels = images.to(device), labels.to(device)
                        optimizer.zero_grad()
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()

                        running_loss += loss.item() * images.size(0)
                        _, predicted = outputs.max(1)
                        correct += predicted.eq(labels).sum().item()
                        total += labels.size(0)

                    epoch_loss = running_loss / total
                    epoch_acc = correct / total

                    print(f"{model_name} | Epoch {epoch+1}/{EPOCHS}, Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}")

                    mlflow.log_metric("loss", epoch_loss, step=epoch)
                    mlflow.log_metric("accuracy", epoch_acc, step=epoch)
                    writer.add_scalar("Loss/train", epoch_loss, epoch)
                    writer.add_scalar("Accuracy/train", epoch_acc, epoch)
                
                writer.close()
                # Save model
                model_path = f"models/{model_name}_lr{lr}_bs{batch_size}.pt"
                os.makedirs("models", exist_ok=True)
                torch.save(model.state_dict(), model_path)
                mlflow.pytorch.log_model(model, "model")
                
                mlflow.log_artifact("metrics.json")

