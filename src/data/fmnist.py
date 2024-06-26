# References:
## https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files
## https://pytorch.org/tutorials/recipes/recipes/custom_dataset_transforms_loader.html

from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor


class FMNIST(Dataset):
    """
    An example custom dataset
    """

    data_raw = {
        "train": "../../data/fashion-mnist_train.csv.zip",
        "test": "../../data/fashion-mnist_test.csv.zip",
    }
    labels = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ]
    sizes = {"input": 0, "output": 0}

    def __init__(self, train: bool = True, transform=ToTensor()):
        pass

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass
