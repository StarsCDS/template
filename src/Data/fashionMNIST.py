import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
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


# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Paths to Zip files relative to the script location
train_zip_file = os.path.join(
    script_dir, "..", "..", "data", "fashion-mnist_train.csv.zip"
)
test_zip_file = os.path.join(
    script_dir, "..", "..", "data", "fashion-mnist_test.csv.zip"
)

# Paths to your CSV files relative to the script location
train_csv_file = os.path.join(script_dir, "..", "..", "data", "fashion-mnist_train.csv")
test_csv_file = os.path.join(script_dir, "..", "..", "data", "fashion-mnist_test.csv")


def extract_zip_if_not_exists(zip_file_path, target_file_path):
    if not os.path.exists(target_file_path):
        with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
            zip_ref.extractall(target_file_path)
        print(f"Files extracted from Zip files")
    else:
        print(f"Files already extracted from Zip files")


# extract the train and the test dataset if they are not already exist
extract_zip_if_not_exists(train_zip_file, train_csv_file)
extract_zip_if_not_exists(test_zip_file, test_csv_file)

# Check if files exist
if not os.path.exists(train_csv_file):
    raise FileNotFoundError(f"Training file not found: {train_csv_file}")
if not os.path.exists(test_csv_file):
    raise FileNotFoundError(f"Test file not found: {test_csv_file}")

# Create custom dataset for train and test datasets
fashion_mnist_train_dataset = FashionMNISTCSV(train_csv_file, transform=None)
fashion_mnist_test_dataset = FashionMNISTCSV(test_csv_file, transform=None)

# Create DataLoader
train_dataloader = DataLoader(fashion_mnist_train_dataset, batch_size=64, shuffle=True)
test_dataloader = DataLoader(fashion_mnist_test_dataset, batch_size=64, shuffle=True)

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


# Function to display images
def show_images(images, labels, file_name):
    fig, axes = plt.subplots(1, len(images), figsize=(12, 12))
    for img, label, ax in zip(images, labels, axes):
        ax.imshow(img.squeeze(), cmap="gray")
        ax.set_title(labels_map[label.item()])
        ax.axis("off")
    plt.savefig(file_name)
    plt.close()


# Get a batch of training images
train_features, train_labels = next(iter(train_dataloader))

# Show a few images with their labels
show_images(train_features[:5], train_labels[:5], "train_image.png")

# Get a batch of test images
test_features, test_labels = next(iter(test_dataloader))

# Show a few test images with their labels
show_images(test_features[:5], test_labels[:5], "test_image.png")
