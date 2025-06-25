# src/train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torch.utils.tensorboard import SummaryWriter
import mlflow
import mlflow.pytorch
import yaml
import json
from preprocess import preprocess_data
# Load hyperparameters from params.yaml
with open("src/params.yaml") as f:
    params = yaml.safe_load(f)

EPOCHS = params["epochs"]
IMG_SIZE = params["img_size"]
LEARNING_RATES = params["lr_list"]
BATCH_SIZES = params["batch_size_list"]

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

# Load preprocessed data
data_dir = "data/processed_data/train"  # DVC-tracked preprocessed data
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3)
])

# Load the dataset from ImageFolder (this assumes the data is organized in class-specific subfolders)
train_dataset = datasets.ImageFolder(data_dir, transform=transform)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mlflow.set_experiment("OxfordPets-MultiModel-Tracking")
print(f"Using device: {device}")
# Training loop for each model
for model_name, builder in model_builders.items():
    for lr in LEARNING_RATES:
        for batch_size in BATCH_SIZES:
            dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            model = builder(len(train_dataset.classes)).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=lr)

            model_file = f"{model_name}_lr{lr}_bs{batch_size}.pt"
            with mlflow.start_run(run_name=f"train_{model_file}"):
                mlflow.log_param("model", model_name)
                mlflow.log_param("epochs", EPOCHS)
                mlflow.log_param("lr", lr)
                mlflow.log_param("batch_size", batch_size)
                writer = SummaryWriter(log_dir=f"logs/train/{model_file}")
                dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(device)
                writer.add_graph(model, dummy_input)
                
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

                # Save model
                model_path = f"models/{model_file}"
                os.makedirs("models", exist_ok=True)
                torch.save(model.state_dict(), model_path)
                
                dummy_input_numpy = dummy_input.cpu().numpy()

                # Log the model with the converted dummy input example
                mlflow.pytorch.log_model(model, name="model", registered_model_name=f"OxfordPetsModel_{model_name}", 
                                         input_example=dummy_input_numpy)
                mlflow.log_artifact(model_path)

                writer.add_embedding(images.view(images.size(0), -1), metadata=labels, tag="embeddings")
                writer.close()