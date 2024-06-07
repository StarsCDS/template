import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def load_data(file_path):
    """Load country data from a CSV file."""
    return pd.read_csv(file_path)

def perform_kmeans_clustering(data, features, num_clusters=3):
    """Perform KMeans clustering on the country data."""
    # Select relevant features
    X = data[features]
    
    # Initialize KMeans algorithm
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    
    # Fit KMeans to the data
    kmeans.fit(X)
    
    # Assign clusters to data points
    data['cluster'] = kmeans.labels_
    
    # Get cluster centers (centroids)
    centroids = kmeans.cluster_centers_
    
    return data, centroids

def save_clustered_data(data, output_file):
    """Save the clustered country data to a CSV file."""
    data.to_csv(output_file, index=False)

def plot_clusters(data, centroids, features, x_feature, y_feature, output_file):
    """Plot the clusters with centroids and save the plot as an image file."""
    plt.figure(figsize=(10, 6))
    
    # Plot data points
    scatter = plt.scatter(data[x_feature], data[y_feature], c=data['cluster'], cmap='viridis', label='Data Points')
    
    # Plot centroids
    plt.scatter(centroids[:, features.index(x_feature)], centroids[:, features.index(y_feature)], marker='x', s=100, color='red', label='Centroids')
    
    plt.xlabel(x_feature)
    plt.ylabel(y_feature)
    plt.title('KMeans Clustering')
    
    # Add legend for data points and centroids
    plt.legend()
    
    # Add color labels on the side
    color_labels = ['Cluster ' + str(i) for i in range(len(centroids))]
    plt.colorbar(scatter, ticks=range(len(centroids)), label='Cluster', orientation='vertical')
    plt.yticks(range(len(centroids)), color_labels)
    
    plt.savefig(output_file)

def main():
    # Path to the country data CSV file
    data_file_path = '/home/harrshini/unsupervised-learning/data/Country-data.csv'
    
    # Features to use for clustering
    features = ['child_mort', 'exports', 'health', 'imports', 'income', 'inflation', 'life_expec', 'total_fer', 'gdpp']
    
    # Number of clusters
    num_clusters = 3
    
    # Load the country data
    country_data = load_data(data_file_path)
    
    # Perform KMeans clustering
    clustered_data, centroids = perform_kmeans_clustering(country_data, features, num_clusters)
    
    # Save the clustered data
    output_file_path = '/home/harrshini/unsupervised-learning/data/clustered_country_data.csv'
    save_clustered_data(clustered_data, output_file_path)
    
    # Plot the clusters with centroids and save the plot as an image file
    plot_output_file1 = '/home/harrshini/unsupervised-learning/data/cluster_plot1.png'
    plot_clusters(clustered_data, centroids, features, 'gdpp', 'child_mort', plot_output_file1)

    # Plot with different features
    plot_output_file2 = '/home/harrshini/unsupervised-learning/data/cluster_plot2.png'
    plot_clusters(clustered_data, centroids, features, 'income', 'life_expec', plot_output_file2)

if __name__ == "__main__":
    main()







