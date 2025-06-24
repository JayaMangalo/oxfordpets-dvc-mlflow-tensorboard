import os
import torch
from torchvision import datasets, transforms
import yaml
from torch.utils.data import random_split
from torchvision.datasets import ImageFolder

def preprocess_data():
    """
    Preprocess the Oxford-IIIT Pet dataset images.
    Resizes and normalizes them, splits into train/test, and saves using capitalized class folders.
    """
    with open("src/params.yaml") as f:
        params = yaml.safe_load(f)

    IMG_SIZE = params["img_size"]
    SEED = params.get("random_seed", 42)
    torch.manual_seed(SEED)

    # Transformations
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ])

    data_dir = "data/oxford-iiit-pet/images"
    dataset = ImageFolder(data_dir, transform=transform)

    # Split
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_data, test_data = random_split(dataset, [train_size, test_size])

    print(f"Processed {len(dataset)} images from {data_dir}")
    print(f"Training data size: {len(train_data)}, Test data size: {len(test_data)}")

    # Save splits
    processed_data_dir = "data/processed_data"
    train_dir = os.path.join(processed_data_dir, "train")
    test_dir = os.path.join(processed_data_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    def save_split(split_data, target_root):
        for idx in range(len(split_data)):
            img_path, label = split_data.dataset.imgs[split_data.indices[idx]]
            filename = os.path.basename(img_path)
            class_name = filename.split("_")[0].capitalize()  # Extract and capitalize

            dest_dir = os.path.join(target_root, class_name)
            os.makedirs(dest_dir, exist_ok=True)

            dest_path = os.path.join(dest_dir, filename)
            try:
                if not os.path.exists(dest_path):
                    os.symlink(os.path.abspath(img_path), dest_path)
            except FileExistsError:
                pass  # Skip if symlink already exists

    save_split(train_data, train_dir)
    save_split(test_data, test_dir)

    return train_data, test_data

if __name__ == "__main__":
    preprocess_data()
