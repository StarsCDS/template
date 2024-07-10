"""
Test suite for the FashionMNIST data module.
"""

import sys
import os
import unittest

# Add the src directory to the Python path
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src"))
)

from data.fmnist import get_data_loaders  # Ensure this import is correct


class TestFashionMNISTDataLoader(unittest.TestCase):
    """
    Test cases for FashionMNIST DataLoader functionality.
    """

    def setUp(self):
        """
        Set up the test environment, including paths and data loaders.
        """
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.train_loader, _ = get_data_loaders(script_dir)

    def test_data_loader(self):
        """
        Test that the DataLoader correctly loads images and labels.
        """
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
