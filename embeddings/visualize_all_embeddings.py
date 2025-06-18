import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

def main(_A: argparse.Namespace):
    """
    이미지 및 텍스트 임베딩을 로드하고 3D PCA로 시각화합니다.
    """
    # --- 1. 데이터 로드 ---
    print("Step 1: Loading embedding files...")
    try:
        with open(_A.image_embeddings_path, 'r') as f:
            image_embeddings_dict = json.load(f)
        print(f"Successfully loaded {len(image_embeddings_dict)} image embeddings.")
    except FileNotFoundError:
        print(f"Error: Image embedding file not found at '{_A.image_embeddings_path}'")
        return

    try:
        with open(_A.text_embeddings_path, 'r') as f:
            text_embeddings_dict = json.load(f)
        print(f"Successfully loaded {len(text_embeddings_dict)} text embeddings.")
    except FileNotFoundError:
        print(f"Error: Text embedding file not found at '{_A.text_embeddings_path}'")
        return

    # --- 2. PCA를 위한 데이터 준비 ---
    print("Step 2: Preparing data for PCA...")
    
    all_embeddings = []
    all_labels = []
    all_types = []

    # ========================== 수정된 부분 시작 ==========================
    # 이미지 임베딩 처리 (형태 통일)
    for emb in image_embeddings_dict.values():
        # emb가 [[...]] 형태의 2차원 리스트인지 확인하고, 그렇다면 1차원 리스트로 변환
        if isinstance(emb, list) and len(emb) == 1 and isinstance(emb[0], list):
            all_embeddings.append(emb[0])
        else:
            # 이미 1차원 리스트인 경우 그대로 추가
            all_embeddings.append(emb)
    # ========================== 수정된 부분 끝 ============================
    
    all_labels.extend(['image'] * len(image_embeddings_dict))
    all_types.extend(['Image'] * len(image_embeddings_dict))

    # 텍스트 임베딩 처리 ([ROOT] 노드 분리)
    root_embedding = None
    for key, value in text_embeddings_dict.items():
        if key == "[ROOT]":
            root_embedding = value
        else:
            all_embeddings.append(value)
            all_labels.append(key)
            all_types.append('Text')
    
    # [ROOT] 노드를 마지막에 추가
    if root_embedding:
        all_embeddings.append(root_embedding)
        all_labels.append('[ROOT]')
        all_types.append('Root')

    print(f"Total embeddings to process: {len(all_embeddings)}")
    
    # NumPy 배열로 변환
    all_embeddings_np = np.array(all_embeddings)
    print(f"Successfully created NumPy array with shape: {all_embeddings_np.shape}")


    # --- 3. 3D PCA 수행 ---
    print("Step 3: Performing 3D PCA...")
    pca = PCA(n_components=3)
    embeddings_3d = pca.fit_transform(all_embeddings_np)
    
    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Total explained variance: {sum(pca.explained_variance_ratio_):.2f}")

    # --- 4. 3D 시각화 ---
    print("Step 4: Generating 3D plot... (You can rotate the plot with your mouse)")
    
    fig = plt.figure(figsize=(_A.fig_width, _A.fig_height), dpi=120)
    ax = fig.add_subplot(111, projection='3d')

    palette = {'Image': 'red', 'Text': 'blue', 'Root': 'black'}
    size_map = {'Image': 20, 'Text': 20, 'Root': 150}
    alpha_map = {'Image': 0.6, 'Text': 0.6, 'Root': 1.0}
    
    for type_name in palette.keys():
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

    try:
        root_index = all_labels.index('[ROOT]')
        ax.text(
            embeddings_3d[root_index, 0], embeddings_3d[root_index, 1], embeddings_3d[root_index, 2],
            '  [ROOT]', fontsize=10, fontweight='bold', color='black'
        )
    except ValueError:
        print("Warning: [ROOT] node not found for labeling.")

    ax.set_title("3D PCA of Image and Text Embeddings", fontsize=16, pad=20)
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_zlabel("Principal Component 3")
    ax.legend(title="Embedding Type")
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize image and text embeddings together using 3D PCA.")
    parser.add_argument("--image-embeddings-path", default='./embeddings/hyperbolic_embeddings.json', help="Path to the image embeddings JSON file.")
    parser.add_argument("--text-embeddings-path", default='./embeddings/text_embeddings.json', help="Path to the text embeddings JSON file.")
    parser.add_argument("--fig-width", type=int, default=12, help="Width of the output plot figure.")
    parser.add_argument("--fig-height", type=int, default=12, help="Height of the output plot figure.")

    _A = parser.parse_args()
    main(_A)