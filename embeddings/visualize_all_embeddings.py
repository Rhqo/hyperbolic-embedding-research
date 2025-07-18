import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# UMAP 라이브러리 임포트 시도
try:
    import umap
except ImportError:
    umap = None

def main(_A: argparse.Namespace):

    # UMAP을 선택했지만 라이브러리가 없는 경우 에러 메시지 출력 후 종료
    if _A.method == 'umap' and umap is None:
        print("UMAP is selected, but the 'umap-learn' library is not installed.")
        print("Please install it using: pip install umap-learn")
        return

    # 인자에 따라 어떤 데이터를 보여줄지 결정
    show_all = not _A.text and not _A.img
    show_images = show_all or _A.img
    show_texts = show_all or _A.text

    all_embeddings = []
    all_labels = []
    all_types = []

    # 이미지 임베딩 처리
    if show_images:
        try:
            with open(_A.image_embeddings_path, 'r') as f:
                image_embeddings_dict = json.load(f)
            for emb in image_embeddings_dict.values():
                if isinstance(emb, list) and len(emb) == 1 and isinstance(emb[0], list):
                    all_embeddings.append(emb[0])
                else:
                    all_embeddings.append(emb)
            all_labels.extend(['image'] * len(image_embeddings_dict))
            all_types.extend(['Image'] * len(image_embeddings_dict))
        except FileNotFoundError:
            print(f"Warning: Image embeddings file not found at '{_A.image_embeddings_path}'")

    # [ROOT] 노드 및 텍스트 임베딩 처리
    root_embedding = None
    try:
        with open(_A.text_embeddings_path, 'r') as f:
            text_embeddings_dict = json.load(f)
        if '[ROOT]' in text_embeddings_dict:
            root_embedding = text_embeddings_dict.pop('[ROOT]')
        if show_texts:
            for key, value in text_embeddings_dict.items():
                all_embeddings.append(value)
                all_labels.append(key)
                all_types.append('Text')
    except FileNotFoundError:
        print(f"Warning: Text embeddings file not found at '{_A.text_embeddings_path}', [ROOT] node cannot be loaded.")
    except json.JSONDecodeError:
        print(f"Warning: Could not decode JSON from '{_A.text_embeddings_path}'")

    if root_embedding:
        all_embeddings.append(root_embedding)
        all_labels.append('[ROOT]')
        all_types.append('Root')

    if not all_embeddings:
        print("No embeddings were loaded. Check your flags or file paths. Exiting.")
        return

    print(f"Total embeddings to process: {len(all_embeddings)}")
    all_embeddings_np = np.array(all_embeddings)

    # PCA or UMAP
    if _A.method == 'umap':
        print("Performing UMAP dimensionality reduction...")
        reducer = umap.UMAP(
            n_components=2,      # 2차원으로 축소
            n_neighbors=15,      # 지역적 구조를 살필 이웃 수 (조정 가능)
            min_dist=0.1,        # 점들이 얼마나 뭉칠지 결정 (조정 가능)
            metric='cosine',     # 임베딩 벡터에는 코사인 유사도가 효과적인 경우가 많음
            random_state=42      # 재현성을 위한 시드값
        )
        embeddings_2d = reducer.fit_transform(all_embeddings_np)
        x_label, y_label = "UMAP Dimension 1", "UMAP Dimension 2"
    else:  # 기본값은 PCA
        print("Performing PCA dimensionality reduction...")
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(all_embeddings_np)
        print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
        print(f"Total explained variance: {sum(pca.explained_variance_ratio_):.2f}")
        x_label, y_label = "Principal Component 1", "Principal Component 2"

    fig, ax = plt.subplots(figsize=(_A.fig_width, _A.fig_height), dpi=120)

    palette = {'Image': 'red', 'Text': 'blue', 'Root': 'black'}
    size_map = {'Image': 20, 'Text': 20, 'Root': 150}
    alpha_map = {'Image': 0.6, 'Text': 0.6, 'Root': 1.0}
    
    for type_name in set(all_types):
        indices = [i for i, t in enumerate(all_types) if t == type_name]
        if not indices: continue
        group_coords = embeddings_2d[indices]
        ax.scatter(
            group_coords[:, 0], group_coords[:, 1],
            c=palette[type_name], s=size_map[type_name],
            alpha=alpha_map[type_name], label=type_name
        )

    if '[ROOT]' in all_labels:
        try:
            root_index = all_labels.index('[ROOT]')
            ax.text(
                embeddings_2d[root_index, 0], embeddings_2d[root_index, 1],
                '  [ROOT]', fontsize=12, fontweight='bold', color='black'
            )
        except (ValueError, IndexError):
            print("Warning: [ROOT] node found in labels but could not be plotted.")

    title_content = []
    if 'Image' in set(all_types): title_content.append('Image')
    if 'Text' in set(all_types): title_content.append('Text')
    
    method_name = _A.method.upper()
    content_str = ' and '.join(title_content)
    plot_title = f"2D {method_name} of {content_str} Embeddings" if content_str else f"2D {method_name} of [ROOT] Embedding"

    ax.set_title(plot_title, fontsize=16, pad=20)
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.legend(title="Embedding Type")
    ax.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize image and/or text embeddings using 2D PCA or UMAP. The [ROOT] node is always displayed if available.")

    parser.add_argument("--image-embeddings-path", default='./embeddings/hyperbolic_embeddings.json', help="Path to the image embeddings JSON file.")
    parser.add_argument("--text-embeddings-path", default='./embeddings/text_embeddings.json', help="Path to the text embeddings JSON file.")
    parser.add_argument("--fig-width", type=int, default=10, help="Width of the output plot figure.")
    parser.add_argument("--fig-height", type=int, default=6, help="Height of the output plot figure.")

    parser.add_argument("--method", type=str, default='pca', choices=['pca', 'umap'], help="Dimensionality reduction method to use: 'pca' or 'umap'.")
    parser.add_argument("--text", action='store_true', help="Visualize only text embeddings (plus the ROOT node).")
    parser.add_argument("--img", action='store_true', help="Visualize only image embeddings (plus the ROOT node).")

    _A = parser.parse_args()
    main(_A)