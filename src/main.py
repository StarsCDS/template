import sys
import os
from typing import Dict, Any, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
<<<<<<< HEAD
<<<<<<< HEAD
=======

>>>>>>> 6b4dd4b (Update README.md)
=======
>>>>>>> b371bf6 (Refactor project structure, implement multiple clustering models, add tests, and synthetic data loader)
from src.config import (
    torch, pd, DataLoader, StandardScaler, BATCH_SIZE
)
from src.model.kmeans import KMeansClusterer
from src.model.dbscan import DBSCANClusterer
from src.model.gmm import GMMClusterer
from src.model.ensemble import EnsembleClusterer
from src.model.agglomerative import AgglomerativeClusterer
from src.model.optics import OPTICSClusterer
from src.model.hdbscan_clusterer import HDBSCANClusterer
from src.data.radar_synthetic import get_dataloader

class ClusteringModelSelector:
    def __init__(self, dataloader: DataLoader):
        self.dataloader = dataloader
        self.models: Dict[str, Any] = {
            'kmeans': KMeansClusterer(),
            'dbscan': DBSCANClusterer(),
            'gmm': GMMClusterer(),
            'ensemble': EnsembleClusterer(),
            'agglomerative': AgglomerativeClusterer(),
            'optics': OPTICSClusterer(),
            'hdbscan': HDBSCANClusterer(),
        }
        self.scaler = StandardScaler()

    def prepare_data(self) -> Tuple[pd.DataFrame, Any]:
        all_data = []
        for batch in self.dataloader:
            all_data.append(batch)
        all_data = torch.cat(all_data, dim=0)
        
        data_np = all_data.numpy()
        
        columns = [
            'Signal Duration (microsec)', 'Azimuthal Angle (degrees)',
            'Elevation Angle (degrees)', 'PRI (microsec)', 'Timestamp (microsec)',
            'Signal Strength (dBm)', 'Signal Frequency (MHz)', 'Amplitude'
        ]
        df = pd.DataFrame(data_np, columns=columns)
        
        features_scaled = self.scaler.fit_transform(data_np)
        
        return df, features_scaled

    def select_model(self, model_name: str) -> None:
        df, features_scaled = self.prepare_data()
        
        if model_name not in self.models:
            raise ValueError(f"Unknown model: {model_name}")
        
        results = self.models[model_name].run(df, features_scaled)

        print(f"Results for {model_name}:")
        if 'scores' in results:
            for metric, score in results['scores'].items():
                print(f"{metric}: {score:.2f}")
        else:
            for metric, score in results.items():
                print(f"{metric}: {score:.2f}")

        if 'optimal_k' in results:
            print(f"Optimal number of clusters: {results['optimal_k']}")

        if 'ensemble_type' in results:
            ensemble_types = {1: "Soft Voting", 2: "Majority Voting", 3: "Stacking"}
            print(f"Ensemble type: {ensemble_types[results['ensemble_type']]}")

def main() -> None:
    dataloader = get_dataloader(batch_size=BATCH_SIZE, shuffle=True)
    model_selector = ClusteringModelSelector(dataloader=dataloader)

    print("Select a clustering model:")
    model_names = list(model_selector.models.keys())
    for i, model_name in enumerate(model_names, start=1):
        print(f"{i}. {model_name}")

    while True:
        choice = input("Enter the number corresponding to your choice: ")
        try:
            choice = int(choice)
            if 1 <= choice <= len(model_names):
                selected_model = model_names[choice - 1]
                break
            print(f"Invalid choice. Please enter a number between 1 and {len(model_names)}")
        except ValueError:
            print("Invalid input. Please enter a number.")

    model_selector.select_model(selected_model)

if __name__ == "__main__":
    main()