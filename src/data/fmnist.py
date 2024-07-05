# References:
## https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files
## https://pytorch.org/tutorials/recipes/recipes/custom_dataset_transforms_loader.html
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import os
import zipfile


class FashionMNISTCSV(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        label = self.data.iloc[idx, 0]
        image = self.data.iloc[idx, 1:].values.astype("float32").reshape(28, 28) / 255.0

        if self.transform:
            image = self.transform(image)

        return torch.tensor(image), label


def extract_zip_if_not_exists(zip_file_path, extract_dir):
    if not os.path.exists(extract_dir):
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)
        print(f"Files extracted from {zip_file_path}")
    else:
        print(f"Files already extracted to {extract_dir}")


def get_data_loaders(script_dir):
    # Paths to Zip files relative to the script location
    train_zip_file = os.path.join(
        script_dir, "..", "..", "data", "fashion-mnist_train.csv.zip"
    )
    test_zip_file = os.path.join(
        script_dir, "..", "..", "data", "fashion-mnist_test.csv.zip"
    )

    # Paths to directories where the CSV files will be extracted
    train_extract_dir = os.path.join(script_dir, "..", "..", "data", "train")
    test_extract_dir = os.path.join(script_dir, "..", "..", "data", "test")

    # Full paths to the CSV files after extraction
    train_csv_file = os.path.join(train_extract_dir, "fashion-mnist_train.csv")
    test_csv_file = os.path.join(test_extract_dir, "fashion-mnist_test.csv")

    # Extract the train and the test dataset if they do not already exist
    extract_zip_if_not_exists(train_zip_file, train_extract_dir)
    extract_zip_if_not_exists(test_zip_file, test_extract_dir)

    # Check if files exist
    if not os.path.exists(train_csv_file):
        raise FileNotFoundError(f"Training file not found: {train_csv_file}")
    if not os.path.exists(test_csv_file):
        raise FileNotFoundError(f"Test file not found: {test_csv_file}")

    # Create custom dataset for train and test datasets
    fashion_mnist_train_dataset = FashionMNISTCSV(train_csv_file, transform=None)
    fashion_mnist_test_dataset = FashionMNISTCSV(test_csv_file, transform=None)

    # Create DataLoader
    train_dataloader = DataLoader(
        fashion_mnist_train_dataset, batch_size=64, shuffle=True
    )
    test_dataloader = DataLoader(
        fashion_mnist_test_dataset, batch_size=64, shuffle=True
    )

    return train_dataloader, test_dataloader


# Define labels for FashionMNIST
labels_map = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot",
}


def show_images(images, labels, file_name):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(images), figsize=(12, 12))
    for img, label, ax in zip(images, labels, axes):
        ax.imshow(img.squeeze(), cmap="gray")
        ax.set_title(labels_map[label.item()])
        ax.axis("off")
    plt.savefig(file_name)
    plt.close()
