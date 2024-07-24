import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import *
from model.kmeans import KMeansClusterer
from model.dbscan import DBSCANClusterer
from model.gmm import GMMClusterer
from model.ensemble import EnsembleClusterer
from model.agglomerative import AgglomerativeClusterer
from model.optics import OPTICSClusterer
from model.hdbscan_clusterer import HDBSCANClusterer
from data.data_loader import get_dataloader
from utils.visualization import plot_ensemble, calculate_clustering_scores


class ClusteringModelSelector:
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.models = {
            'kmeans': KMeansClusterer(),
            'dbscan': DBSCANClusterer(),
            'gmm': GMMClusterer(),
            'ensemble': EnsembleClusterer(),
            'agglomerative': AgglomerativeClusterer(),
            'optics': OPTICSClusterer(),
            'hdbscan': HDBSCANClusterer(),
        }
        self.scaler = StandardScaler()
        self.output_dir = os.path.join(os.path.dirname(__file__), 'model', 'plots')

    def prepare_data(self):
        # Collect all data from the dataloader into a single tensor
        all_data = []
        for batch in self.dataloader:
            all_data.append(batch)
        all_data = torch.cat(all_data, dim=0)
        
        # Convert the tensor to a numpy array
        data_np = all_data.numpy()
        
        # Create a DataFrame
        columns = ['Signal Duration (microsec)', 'Azimuthal Angle (degrees)', 'Elevation Angle (degrees)',
                   'PRI (microsec)', 'Timestamp (microsec)', 'Signal Strength (dBm)', 'Signal Frequency (MHz)', 'Amplitude']
        df = pd.DataFrame(data_np, columns=columns)
        
        # Scale the data
        features_scaled = self.scaler.fit_transform(data_np)
        
        return df, features_scaled

    def select_model(self, model_name):
        df, features_scaled = self.prepare_data()
        
        # Run the selected model
        if model_name not in self.models:
            raise ValueError(f"Unknown model: {model_name}")
        
        results = self.models[model_name].run(df, features_scaled, self.output_dir)

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

def main():
    # Get the dataloader
    dataloader = get_dataloader(batch_size=32, shuffle=True)

    # Initialize the model selector
    model_selector = ClusteringModelSelector(dataloader=dataloader)

    # Prompt the user to select a model
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
            else:
                print("Invalid choice. Please enter a number between 1 and", len(model_names))
        except ValueError:
            print("Invalid input. Please enter a number.")

    # Run the selected model
    model_selector.select_model(selected_model)

if __name__ == "__main__":
    main()