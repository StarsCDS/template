"""
Synthetic Radar Data Generator and Loader Module.

This module provides functionality to generate synthetic radar data
and load it using PyTorch's Dataset and DataLoader classes.

Classes:
    RadarDataset: A custom Dataset class for synthetic radar data.

Functions:
    get_dataloader: Creates and returns a DataLoader for the RadarDataset.
"""

from typing import List
from src.config import (
    np, pd, torch, Dataset, DataLoader, datetime, timedelta
)
class RadarDataset(Dataset):
    """
    A custom Dataset class for synthetic radar data.

    This class generates synthetic radar data and provides methods
    to access it as a PyTorch Dataset.

    Attributes:
        data (pd.DataFrame): The generated synthetic radar data.
    """

    def __init__(self, n_samples: int = 1000):
        """
        Initialize the RadarDataset.

        Args:
            n_samples (int): Number of samples to generate. Default is 1000.
        """
        self.data = self.generate_radar_data(n_samples)

    def __len__(self) -> int:
        """
        Get the number of samples in the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Get a sample from the dataset.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            torch.Tensor: The sample at the given index.
        """
        return torch.tensor(self.data.iloc[idx].values, dtype=torch.float32)

    @staticmethod
    def generate_radar_data(n_samples: int = 1000) -> pd.DataFrame:
        """
        Generate synthetic radar data.

        Args:
            n_samples (int): Number of samples to generate. Default is 1000.

        Returns:
            pd.DataFrame: A DataFrame containing the generated radar data.
        """
        np.random.seed(42)  # For reproducibility of the results

        signal_duration = np.random.uniform(1e-6, 1e-3, n_samples) * 1e6
        azimuthal_angle = np.random.uniform(0, 360, n_samples)
        elevation_angle = np.random.uniform(-90, 90, n_samples)
        pri = np.random.uniform(1e-3, 1, n_samples) * 1e6
        start_time = datetime.now()
        timestamps: List[float] = [
            start_time + timedelta(microseconds=int(x))
            for x in np.cumsum(np.random.uniform(0, 1000, n_samples))
        ]
        timestamps = [(t - start_time).total_seconds() * 1e6 for t in timestamps]
        signal_strength = np.random.uniform(-100, 0, n_samples)
        signal_frequency = np.random.uniform(30, 30000, n_samples)
        amplitude = np.random.uniform(0, 10, n_samples)

        df = pd.DataFrame({
            'Signal Duration (microsec)': signal_duration,
            'Azimuthal Angle (degrees)': azimuthal_angle,
            'Elevation Angle (degrees)': elevation_angle,
            'PRI (microsec)': pri,
            'Timestamp (microsec)': timestamps,
            'Signal Strength (dBm)': signal_strength,
            'Signal Frequency (MHz)': signal_frequency,
            'Amplitude': amplitude
        })

        return df

def get_dataloader(batch_size: int = 32, shuffle: bool = True, num_workers: int = 4) -> DataLoader:
    """
    Create a DataLoader for the RadarDataset.

    Args:
        batch_size (int): Number of samples per batch. Default is 32.
        shuffle (bool): Whether to shuffle the data. Default is True.
        num_workers (int): Number of subprocesses to use for data loading. Default is 4.

    Returns:
        DataLoader: A PyTorch DataLoader for the RadarDataset.
    """
    
    dataset = RadarDataset()
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)