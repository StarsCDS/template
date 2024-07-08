import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


# Load the dataset
file_path = 'synthetic_radar_data.csv'
df = pd.read_csv(file_path)

# Generate synthetic data with 5 features
X, y = make_blobs(n_samples=1000, n_features=8, centers=5, cluster_std=1.0, random_state=42)

# Convert data to DataFrame
df = pd.DataFrame(X, columns=['Signal Frequency (MHz)', 'Elevation Angle (degrees)', 'Azimuth Angle (degrees)',
                              'Signal Duration (microsec)', 'Signal Strength (dBm)', 'PRI (microsec)',
                              'Doppler Frequency', 'Amplitude'])

# Standardize the data
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# Apply PCA to capture all components (for visualization purposes)
pca = PCA(n_components=3, random_state=42)  # Reduce to 3 components for 3D visualization
pca_components = pca.fit_transform(df_scaled)

# Initialize KMeans with k=5
kmeans = KMeans(n_clusters=5, random_state=42)
df['Cluster'] = kmeans.fit_predict(df_scaled)

# Visualize the clusters with a 3D scatter plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot with color encoding by cluster
scatter = ax.scatter(df['Signal Frequency (MHz)'], df['Elevation Angle (degrees)'],
                     df['Azimuth Angle (degrees)'], c=df['Cluster'], cmap='viridis', marker='o')

# Set labels and title for the plot
ax.set_xlabel('Signal Frequency (MHz)')
ax.set_ylabel('Elevation Angle (degrees)')
ax.set_zlabel('Azimuth Angle (degrees)')
ax.set_title('3D Scatter Plot with K-Means Clustering (Original Features)')

# Add color bar which maps values to colors
fig.colorbar(scatter, ax=ax, label='Cluster')

# Show the plot
plt.show()