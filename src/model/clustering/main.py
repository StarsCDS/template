# src/model/clustering/main.py
from run_kmeans import run_kmeans
from run_dbscan import run_dbscan
from run_gmm import run_gmm
from run_ensemble import run_ensemble
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from data.data_loader import get_dataloader

class ClusteringModelSelector:
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.models = {
            'kmeans': run_kmeans,
            'dbscan': run_dbscan,
            'gmm': run_gmm,
            'ensemble': run_ensemble,
        }
        self.scaler = StandardScaler()

    def select_model(self, model_name, **kwargs):
        # Collect all data from the dataloader into a single tensor
        all_data = []
        for batch in self.dataloader:
            all_data.append(batch[0])
        all_data = torch.cat(all_data, dim=0)

        # Convert the tensor to a numpy array
        data_np = all_data.numpy()

        # Scale the data
        features_scaled = self.scaler.fit_transform(data_np)

        model_function = self.models[model_name]
        results = model_function(data_np, features_scaled, **kwargs)
        print(f"Results for {model_name} with parameters {kwargs}:")
        for metric, score in results.items():
            print(f"{metric}: {score:.2f}")


# Get the dataloader
dataloader = get_dataloader(csv_file, batch_size=32, shuffle=True)

# Initialize the model selector
model_selector = ClusteringModelSelector(dataloader=dataloader)

# Prompt the user to select a model
print("Select a clustering model:")
for i, model_name in enumerate(model_selector.models.keys(), start=1):
    print(f"{i}. {model_name}")

choice = int(input("Enter the number corresponding to your choice: "))
selected_model = list(model_selector.models.keys())[choice - 1]

# Run the selected model with the corresponding parameters
if selected_model == 'kmeans':
    model_selector.select_model('kmeans', n_clusters=3, random_state=42)
elif selected_model == 'dbscan':
    model_selector.select_model('dbscan', eps=0.5, min_samples=5)
elif selected_model == 'gmm':
    model_selector.select_model('gmm', n_components=3, random_state=42)
elif selected_model == 'ensemble':
    model_selector.select_model('ensemble', n_clusters=3)