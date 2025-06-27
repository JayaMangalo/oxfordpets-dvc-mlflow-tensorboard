import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import accuracy_score
import mlflow
from torch.utils.tensorboard import SummaryWriter
import yaml

# Load hyperparameters
with open("src/params.yaml") as f:
    params = yaml.safe_load(f)

IMG_SIZE = params["img_size"]

# Define image transforms
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3)
])

# Load test dataset
test_data_dir = "data/processed_data/test"
test_dataset = datasets.ImageFolder(test_data_dir, transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Model builders
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
    model = models.mobilenet_v2(pretrained=False)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    return model

def build_resnet18(num_classes):
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

model_builders = {
    "simple_cnn": build_simple_cnn,
    "mobilenet_v2": build_mobilenet,
    "resnet18": build_resnet18
}


experiment_name = "OxfordPets-MultiModel-Tracking"
try:
    mlflow.set_experiment(experiment_name)  # This will create the experiment if it doesn't exist
    print(f"Experiment '{experiment_name}' is set or created.")
except mlflow.exceptions.MlflowException as e:
    print(f"Error setting experiment: {e}")
    raise

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Evaluate all models in the models/ directory
model_dir = "models"
model_files = [f for f in os.listdir(model_dir) if f.endswith(".pt")]

results = []
    
for model_file in model_files:
    model_path = os.path.join(model_dir, model_file)

    # Parse model name from file
    model_name = model_file.split("_lr")[0]
    if model_name not in model_builders:
        print(f"Skipping {model_file}: unknown model type '{model_name}'")
        continue

    # Load the correct model
    model = model_builders[model_name](len(test_dataset.classes)).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)

    print(f"Model {model_file} | Test accuracy: {accuracy:.4f}")
    results.append((model_file, accuracy))
    with mlflow.start_run(run_name=f"evaluation_{model_file}"):
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("model_file", model_file)
        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.pytorch.log_model(model, "model", registered_model_name=f"OxfordPetsModel_{model_name}")

        writer = SummaryWriter(log_dir=f"logs/evaluation/{model_file}")
        writer.add_scalar("Accuracy/test", accuracy, 0)
        writer.add_text("Model Info", f"Evaluated {model_file}", 0)
        writer.close()

# Save to json
import json


os.makedirs("metrics", exist_ok=True)
results_path = "metrics/results.json"
with open(results_path, "w") as f:
    json.dump(results, f, indent=4)
print(f"Evaluation results saved to {results_path}")
