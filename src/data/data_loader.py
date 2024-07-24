# data_loader.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import *

class RadarDataset(Dataset):
    def __init__(self, n_samples=1000):
        self.data = self.generate_radar_data(n_samples)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data.iloc[idx].values, dtype=torch.float32)

    @staticmethod
    def generate_radar_data(n_samples=1000):
        np.random.seed(42)  # For reproducibility of the results

        signal_duration = np.random.uniform(1e-6, 1e-3, n_samples) * 1e6
        azimuthal_angle = np.random.uniform(0, 360, n_samples)
        elevation_angle = np.random.uniform(-90, 90, n_samples)
        pri = np.random.uniform(1e-3, 1, n_samples) * 1e6
        start_time = datetime.now()
        timestamps = [start_time + timedelta(microseconds=int(x)) for x in np.cumsum(np.random.uniform(0, 1000, n_samples))]
        timestamps = [(t - start_time).total_seconds() * 1e6 for t in timestamps]  # Convert to microseconds
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

def get_dataloader(batch_size=32, shuffle=True, num_workers=4):
    dataset = RadarDataset()
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)