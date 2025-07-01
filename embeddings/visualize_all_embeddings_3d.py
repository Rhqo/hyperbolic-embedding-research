import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

# Attempt to import UMAP library
try:
    import umap
except ImportError:
    umap = None

def main(_A: argparse.Namespace):

    # If UMAP is selected but the library isn't found, print an error and exit
    if _A.method == 'umap' and umap is None:
        print("UMAP is selected, but the 'umap-learn' library is not installed.")
        print("Please install it using: pip install umap-learn")
        return

    # Determine which data to display based on arguments
    show_all = not _A.text and not _A.img
    show_images = show_all or _A.img
    show_texts = show_all or _A.text

    all_embeddings = []
    all_labels = []
    all_types = []

    # Process image embeddings
    if show_images:
        try:
            with open(_A.image_embeddings_path, 'r') as f:
                image_embeddings_dict = json.load(f)
            
            for emb in image_embeddings_dict.values():
                # Standardize embedding format (handle nested lists)
                if isinstance(emb, list) and len(emb) == 1 and isinstance(emb[0], list):
                    all_embeddings.append(emb[0])
                else:
                    all_embeddings.append(emb)
            
            all_labels.extend(['image'] * len(image_embeddings_dict))
            all_types.extend(['Image'] * len(image_embeddings_dict))
        except FileNotFoundError:
            print(f"Warning: Image embeddings file not found at '{_A.image_embeddings_path}'")

    # Process [ROOT] node and text embeddings
    root_embedding = None
    try:
        with open(_A.text_embeddings_path, 'r') as f:
            text_embeddings_dict = json.load(f)

        # Separate the [ROOT] node from the dictionary (pop removes it from the original)
        if '[ROOT]' in text_embeddings_dict:
            root_embedding = text_embeddings_dict.pop('[ROOT]')

        # Process the remaining text embeddings based on the show_texts flag
        if show_texts:
            for key, value in text_embeddings_dict.items():
                all_embeddings.append(value)
                all_labels.append(key)
                all_types.append('Text')

    except FileNotFoundError:
        # If the text file is not found, the ROOT is also unavailable, so just show a warning
        print(f"Warning: Text embeddings file not found at '{_A.text_embeddings_path}', [ROOT] node cannot be loaded.")
    except json.JSONDecodeError:
        print(f"Warning: Could not decode JSON from '{_A.text_embeddings_path}'")

    # After processing all other embeddings, add the separated [ROOT] node to the end of the list
    if root_embedding:
        all_embeddings.append(root_embedding)
        all_labels.append('[ROOT]')
        all_types.append('Root')

    # If no embeddings are loaded, exit
    if not all_embeddings:
        print("No embeddings were loaded. Check your flags or file paths. Exiting.")
        return

    print(f"Total embeddings to process: {len(all_embeddings)}")
    all_embeddings_np = np.array(all_embeddings)

    # --- Dimensionality Reduction (PCA vs UMAP) ---
    # Perform dimensionality reduction based on the user's chosen method
    if _A.method == 'umap':
        print("Performing UMAP dimensionality reduction to 3 components...")
        reducer = umap.UMAP(
            n_components=3,      # Reduce to 3 components
            n_neighbors=15,      # Number of neighbors to consider for local structure (tunable)
            min_dist=0.1,        # How tightly to pack points together (tunable)
            metric='minkowski',     # Cosine similarity is often effective for embedding vectors
            random_state=42      # Seed for reproducibility
        )
        embeddings_3d = reducer.fit_transform(all_embeddings_np)
        x_label, y_label, z_label = "UMAP Dimension 1", "UMAP Dimension 2", "UMAP Dimension 3"
    else:  # Default is PCA
        print("Performing PCA dimensionality reduction to 3 components...")
        pca = PCA(n_components=3)
        embeddings_3d = pca.fit_transform(all_embeddings_np)
        print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
        print(f"Total explained variance: {sum(pca.explained_variance_ratio_):.2f}")
        x_label, y_label, z_label = "Principal Component 1", "Principal Component 2", "Principal Component 3"

    # --- 3D Plotting ---
    fig = plt.figure(figsize=(_A.fig_width, _A.fig_height), dpi=120)
    ax = fig.add_subplot(111, projection='3d')

    palette = {'Image': 'red', 'Text': 'blue', 'Root': 'black'}
    size_map = {'Image': 20, 'Text': 20, 'Root': 150}
    alpha_map = {'Image': 0.6, 'Text': 0.6, 'Root': 1.0}
    
    # Iterate over the actual loaded types using set(all_types)
    for type_name in set(all_types):
        indices = [i for i, t in enumerate(all_types) if t == type_name]
        if not indices:
            continue
        
        group_coords = embeddings_3d[indices]
        ax.scatter(
            group_coords[:, 0], group_coords[:, 1], group_coords[:, 2],
            c=palette[type_name],
            s=size_map[type_name],
            alpha=alpha_map[type_name],
            label=type_name,
            depthshade=True
        )

    # Attempt to label the [ROOT] node only if it was loaded
    if '[ROOT]' in all_labels:
        try:
            root_index = all_labels.index('[ROOT]')
            ax.text(
                embeddings_3d[root_index, 0], embeddings_3d[root_index, 1], embeddings_3d[root_index, 2],
                '  [ROOT]', fontsize=10, fontweight='bold', color='black'
            )
        except (ValueError, IndexError):
            print("Warning: [ROOT] node found in labels but could not be plotted.")

    # --- Dynamic Title and Labels ---
    title_content = []
    if 'Image' in set(all_types):
        title_content.append('Image')
    if 'Text' in set(all_types):
        title_content.append('Text')
    
    method_name = _A.method.upper()
    content_str = ' and '.join(title_content)
    plot_title = f"3D {method_name} of {content_str} Embeddings" if content_str else f"3D {method_name} of [ROOT] Embedding"

    ax.set_title(plot_title, fontsize=16, pad=20)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)
    ax.legend(title="Embedding Type")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize image and/or text embeddings in 3D using PCA or UMAP. The [ROOT] node is always displayed if available.")
    
    # File Path Arguments
    parser.add_argument("--image-embeddings-path", default='./embeddings/hyperbolic_embeddings.json', help="Path to the image embeddings JSON file.")
    parser.add_argument("--text-embeddings-path", default='./embeddings/text_embeddings.json', help="Path to the text embeddings JSON file.")
    
    # Plotting Arguments
    parser.add_argument("--fig-width", type=int, default=10, help="Width of the output plot figure.")
    parser.add_argument("--fig-height", type=int, default=6, help="Height of the output plot figure.")
    
    # Control Arguments
    parser.add_argument("--method", type=str, default='pca', choices=['pca', 'umap'], help="Dimensionality reduction method to use: 'pca' or 'umap'.")
    parser.add_argument("--text", action='store_true', help="Visualize only text embeddings (plus the ROOT node).")
    parser.add_argument("--img", action='store_true', help="Visualize only image embeddings (plus the ROOT node).")

    _A = parser.parse_args()
    main(_A)