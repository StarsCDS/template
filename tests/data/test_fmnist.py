import sys
import os
import unittest

# Add the src directory to the Python path
sys.path.insert(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from data.fmnist import get_data_loaders  # Adjust the import as needed


class TestFashionMNISTDataLoader(unittest.TestCase):
    def setUp(self):
        # Assuming get_data_loaders is a function that returns the train and test DataLoaders
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.train_loader, _ = get_data_loaders(script_dir)

    def test_data_loader(self):
        for images, labels in self.train_loader:
            self.assertEqual(
                images.shape[1:], (28, 28), "Image dimensions are incorrect"
            )
            self.assertEqual(
                labels.shape[0],
                images.shape[0],
                "Mismatch in batch size of images and labels",
            )
            break  # Only check the first batch for simplicity


if __name__ == "__main__":
    unittest.main()
