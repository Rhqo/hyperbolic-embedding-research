import json
import argparse
import numpy as np
import torch
from PIL import Image

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from torchvision import transforms as T

from meru import lorentz as L
from meru.config import LazyConfig, LazyFactory
from meru.models import MERU, CLIPBaseline
from meru.utils.checkpointing import CheckpointManager


def interpolate(model, feats: torch.Tensor, root_feat: torch.Tensor, steps: int):
    # MERU의 경우, 원점의 탄젠트 공간에서 선형 보간 수행
    if isinstance(model, MERU):
        # log_map0: 쌍곡 공간의 점을 원점의 탄젠트 공간(유클리드 공간)으로 매핑
        feats = L.log_map0(feats, model.curv.exp())

    interp_feats = [
        torch.lerp(root_feat, feats, weight.item())
        for weight in torch.linspace(0.0, 1.0, steps=steps)
    ]
    interp_feats = torch.stack(interp_feats)

    # MERU의 경우 exp_map0을 통해 다시 쌍곡 공간으로 리프팅, CLIP은 L2 정규화
    if isinstance(model, MERU):
        interp_feats = L.exp_map0(interp_feats, model.curv.exp())
    else:
        interp_feats = torch.nn.functional.normalize(interp_feats, dim=-1)

    # (이미지 -> ROOT) 순서로 경로를 뒤집음
    return interp_feats.flip(0)


def calc_scores(model, image_feats: torch.Tensor, text_feats: torch.Tensor, has_root: bool):
    if isinstance(model, MERU):
        # 쌍곡 내적을 사용하여 유사도 계산
        scores = L.pairwise_inner(image_feats, text_feats, model.curv.exp())
        _aper = L.half_aperture(text_feats, model.curv.exp())
        _oxy_angle = L.oxy_angle(
            text_feats[:, None, :], image_feats[None, :, :], model.curv.exp()
        )
        entailment_energy = _oxy_angle - _aper[..., None]
        if has_root:
            entailment_energy[-1, ...] = 0
        scores[entailment_energy.T > 0] = -1e12
        return scores
    else:
        # CLIP의 경우, 코사인 유사도를 위해 내적 사용
        return image_feats @ text_feats.T

@torch.inference_mode()
def main(_A: argparse.Namespace):
    # --- 1. 설정 및 모델/데이터 로드 ---
    print("Step 1: Setting up and loading model/data...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # MERU 모델 로드
    _C_TRAIN = LazyConfig.load(_A.train_config)
    model = LazyFactory.build_model(_C_TRAIN, device).eval()
    CheckpointManager(model=model).load(_A.checkpoint_path)

    # 미리 생성된 text_embedding.json 로드
    with open(_A.text_embeddings_path, 'r') as f:
        text_embeddings_dict = json.load(f)
    
    text_pool = list(text_embeddings_dict.keys())
    text_feats_pool = torch.tensor(list(text_embeddings_dict.values()), device=device)
    root_feat = torch.tensor(text_embeddings_dict['[ROOT]'], device=device)

    # --- 2. 이미지 임베딩 생성 ---
    print(f"Step 2: Generating embedding for '{_A.image_path}'...")
    image = Image.open(_A.image_path).convert("RGB")
    image_transform = T.Compose([
        T.Resize(224, T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor()
    ])
    image_tensor = image_transform(image).to(device)
    image_feats = model.encode_image(image_tensor[None, ...], project=True)[0]

    # --- 3. Traversal 경로 찾기 ---
    print("Step 3: Finding the traversal path from [IMAGE] to [ROOT]...")
    interp_feats = interpolate(model, image_feats, root_feat, _A.steps)
    
    # 보간된 각 지점에서 가장 가까운 텍스트 찾기
    scores = calc_scores(model, interp_feats, text_feats_pool, has_root=True)
    _, nn1_idxs = scores.max(dim=-1)
    
    # 중복 제거된 경로 생성
    traversal_texts = []
    for idx in nn1_idxs:
        text = text_pool[idx.item()]
        if text not in traversal_texts:
            traversal_texts.append(text)
    
    print("\nTraversal Path Found:")
    print(" -> ".join(traversal_texts))

    # --- 4. PCA를 위한 데이터 준비 및 차원 축소 ---
    print("\nStep 4: Performing PCA for visualization...")
    # 시각화할 모든 임베딩: 이미지 + 전체 텍스트 풀
    all_embeddings = torch.cat([image_feats.unsqueeze(0), text_feats_pool]).cpu().numpy()
    
    # 라벨 및 타입 설정
    all_labels = ["[IMAGE]"] + text_pool
    all_types = ['Image'] + ['Background'] * (len(text_pool) - 1) + ['Root']
    
    # Traversal 경로에 있는 노드들의 타입 변경
    for i, label in enumerate(all_labels):
        if label in traversal_texts and label not in ["[IMAGE]", "[ROOT]"]:
            all_types[i] = 'Traversal Node'

    # PCA 수행
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(all_embeddings)

    # --- 5. 시각화 ---
    print("Step 5: Generating plot...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(_A.fig_width, _A.fig_height), dpi=120)
    
    palette = {
        'Image': 'red',
        'Traversal Node': 'darkorange',
        'Root': 'black',
        'Background': 'lightgray'
    }
    sizes = {
        'Image': 200,
        'Traversal Node': 100,
        'Root': 200,
        'Background': 20
    }

    # 전체 노드 Scatter Plot
    sns.scatterplot(
        x=embeddings_2d[:, 0],
        y=embeddings_2d[:, 1],
        hue=all_types,
        size=all_types,
        sizes=sizes,
        palette=palette,
        alpha=0.8,
        ax=ax,
        legend='full'
    )
    
    # Traversal 경로 추출 및 연결
    path_indices = [all_labels.index(label) for label in traversal_texts]
    path_coords = embeddings_2d[path_indices]

    ax.plot(
        path_coords[:, 0],
        path_coords[:, 1],
        color='gray',
        linestyle='--',
        linewidth=1.5,
        marker='o',
        markersize=0, # 마커는 scatter에 있으므로 선만 그림
        zorder=0 # 점 아래에 선이 그려지도록 설정
    )

    # 경로 노드에 텍스트 라벨 추가
    for i, label in enumerate(all_labels):
        if all_types[i] != 'Background':
            ax.text(
                embeddings_2d[i, 0] + 0.05,
                embeddings_2d[i, 1] + 0.05,
                label,
                fontsize=9,
                fontweight='bold'
            )

    ax.set_title(f"PCA Visualization of Traversal: '{_A.image_path.split('/')[-1]}'", fontsize=16)
    ax.set_xlabel("Principal Component 1", fontsize=12)
    ax.set_ylabel("Principal Component 2", fontsize=12)
    ax.legend(title="Node Type")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize image-to-text traversal using PCA.")
    # --- 파일 경로 설정 ---
    parser.add_argument("--image-path", default='./assets/green_dress.jpeg', help="Path to an image for performing traversal.")
    parser.add_argument("--text-embeddings-path", default='./embeddings/text_embeddings.json', help="Path to the pre-generated text embeddings JSON file.")
    parser.add_argument("--checkpoint-path", default='./checkpoints/meru_vit_b.pth', help="Path to the MERU model checkpoint.")
    parser.add_argument("--train-config", default='./configs/train_meru_vit_b.py', help="Path to the model's training config file.")
    
    # --- 시각화 옵션 ---
    parser.add_argument("--steps", type=int, default=50, help="Number of interpolation steps for traversal.")
    parser.add_argument("--fig-width", type=int, default=14, help="Width of the output plot figure.")
    parser.add_argument("--fig-height", type=int, default=10, help="Height of the output plot figure.")

    _A = parser.parse_args()
    main(_A)