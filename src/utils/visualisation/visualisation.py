from src.config import *


def get_embeddings(model, dataloader, device):
    model.eval()
    embeddings = []
    labels = []
    with torch.no_grad():
        for anchor, _, _, batch_labels in tqdm(dataloader, desc="Generating embeddings"):
            images = anchor.to(device)
            outputs = model.forward_once(images)
            embeddings.append(outputs.cpu().numpy())
            labels.extend(batch_labels.numpy())

    embeddings = np.vstack(embeddings)
    labels = np.array(labels)
    return embeddings, labels

def visualize_embeddings(embeddings, labels, title, save_path=None):
    # Reduce number of samples if too large
    max_samples = 500
    if len(embeddings) > max_samples:
        indices = np.random.choice(len(embeddings), max_samples, replace=False)
        embeddings = embeddings[indices]
        labels = labels[indices]

    print(f"Performing t-SNE for {title}...")
    tsne = TSNE(n_components=2, random_state=42, verbose=1)
    embeddings_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap='tab10')
    plt.colorbar(scatter, ticks=range(5))  # Assuming 5 classes
    plt.title(title)
    plt.xlabel('t-SNE feature 1')
    plt.ylabel('t-SNE feature 2')

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()




def visualize_embeddings_umap(embeddings, labels, title):
    n_samples = min(len(embeddings), len(labels))
    embeddings = embeddings[:n_samples]
    labels = labels[:n_samples]

    print("Performing UMAP...")
    umap = UMAP(n_components=2, random_state=42)
    embeddings_2d = umap.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap='tab10')
    plt.colorbar(scatter)
    plt.title(title)
    plt.xlabel('UMAP feature 1')
    plt.ylabel('UMAP feature 2')
    plt.show()