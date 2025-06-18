import json
import argparse
import numpy as np
import torch
from PIL import Image

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 3D 시각화를 위한 임포트
import umap
from torchvision import transforms as T

# MERU 라이브러리 및 관련 모듈 임포트
from meru import lorentz as L
from meru.config import LazyConfig, LazyFactory
from meru.models import MERU, CLIPBaseline
from meru.utils.checkpointing import CheckpointManager

# 경고 메시지 무시 (선택 사항)
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def interpolate(model, feats: torch.Tensor, root_feat: torch.Tensor, steps: int):
    """주어진 특징 벡터와 `[ROOT]` 사이를 모델 유형에 따라 보간합니다."""
    if isinstance(model, MERU):
        feats = L.log_map0(feats, model.curv.exp())
    interp_feats = [
        torch.lerp(root_feat, feats, weight.item())
        for weight in torch.linspace(0.0, 1.0, steps=steps)
    ]
    interp_feats = torch.stack(interp_feats)
    if isinstance(model, MERU):
        interp_feats = L.exp_map0(interp_feats, model.curv.exp())
    else:
        interp_feats = torch.nn.functional.normalize(interp_feats, dim=-1)
    return interp_feats.flip(0)


def calc_scores(model, image_feats: torch.Tensor, text_feats: torch.Tensor, has_root: bool):
    """이미지와 텍스트 특징 간의 유사도 점수를 계산합니다."""
    if isinstance(model, MERU):
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
        return image_feats @ text_feats.T

@torch.inference_mode()
def main(_A: argparse.Namespace):
    # --- 1, 2, 3 단계는 2D 버전과 동일 ---
    print("Steps 1-3: Loading data, generating embeddings, and finding traversal path...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _C_TRAIN = LazyConfig.load(_A.train_config)
    model = LazyFactory.build_model(_C_TRAIN, device).eval()
    CheckpointManager(model=model).load(_A.checkpoint_path)
    with open(_A.text_embeddings_path, 'r') as f:
        text_embeddings_dict = json.load(f)
    text_pool = list(text_embeddings_dict.keys())
    text_feats_pool = torch.tensor(list(text_embeddings_dict.values()), device=device)
    root_feat = torch.tensor(text_embeddings_dict['[ROOT]'], device=device)
    image = Image.open(_A.image_path).convert("RGB")
    image_transform = T.Compose([
        T.Resize(224, T.InterpolationMode.BICUBIC), T.CenterCrop(224), T.ToTensor()
    ])
    image_tensor = image_transform(image).to(device)
    image_feats = model.encode_image(image_tensor[None, ...], project=True)[0]
    interp_feats = interpolate(model, image_feats, root_feat, _A.steps)
    scores = calc_scores(model, interp_feats, text_feats_pool, has_root=True)
    _, nn1_idxs = scores.max(dim=-1)
    traversal_texts = []
    for idx in nn1_idxs:
        text = text_pool[idx.item()]
        if text not in traversal_texts:
            traversal_texts.append(text)
    print("\nTraversal Path Found:")
    print(" -> ".join(traversal_texts))

    # --- 4. 3D UMAP을 위한 데이터 준비 및 차원 축소 ---
    print(f"\nStep 4: Performing 3D UMAP (metric: '{_A.metric}') for visualization...")
    all_embeddings = torch.cat([image_feats.unsqueeze(0), text_feats_pool]).cpu().numpy()
    all_labels = ["[IMAGE]"] + text_pool
    all_types = ['Image'] + ['Background'] * (len(text_pool) - 1) + ['Root']
    for i, label in enumerate(all_labels):
        if label in traversal_texts and label not in ["[IMAGE]", "[ROOT]"]:
            all_types[i] = 'Traversal Node'

    # UMAP 감속기(reducer) 생성: n_components=3으로 변경
    reducer = umap.UMAP(
        n_neighbors=_A.n_neighbors,
        min_dist=_A.min_dist,
        metric=_A.metric,
        n_components=3,  # 3차원으로 설정
        random_state=42
    )
    embeddings_3d = reducer.fit_transform(all_embeddings)

    # --- 5. 3D 시각화 ---
    print("Step 5: Generating 3D plot... (You can rotate the plot with your mouse)")
    
    fig = plt.figure(figsize=(_A.fig_width, _A.fig_height), dpi=120)
    ax = fig.add_subplot(111, projection='3d') # 3D 축 생성

    palette = {'Image': 'red', 'Traversal Node': 'darkorange', 'Root': 'black', 'Background': 'lightgray'}
    size_map = {'Image': 250, 'Traversal Node': 120, 'Root': 250, 'Background': 25}
    
    # 타입에 따라 색상과 크기 리스트 생성
    colors = [palette[t] for t in all_types]
    sizes = [size_map[t] for t in all_types]
    
    # 3D Scatter Plot
    x, y, z = embeddings_3d[:, 0], embeddings_3d[:, 1], embeddings_3d[:, 2]
    ax.scatter(x, y, z, c=colors, s=sizes, alpha=0.7, depthshade=True)

    # Traversal 경로 추출 및 3D 라인으로 연결
    path_indices = [all_labels.index(label) for label in traversal_texts]
    path_coords = embeddings_3d[path_indices]
    ax.plot(
        path_coords[:, 0], path_coords[:, 1], path_coords[:, 2],
        color='gray', linestyle='--', linewidth=2, marker='o', markersize=0, zorder=10
    )

    # 경로 노드에 3D 텍스트 라벨 추가
    for i, label in enumerate(all_labels):
        if all_types[i] != 'Background':
            ax.text(
                embeddings_3d[i, 0], embeddings_3d[i, 1], embeddings_3d[i, 2],
                f'  {label}', fontsize=9, fontweight='bold', zorder=11,
                color=palette[all_types[i]]
            )

    # 수동으로 범례 생성
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', label=k, 
                   markerfacecolor=v, markersize=np.sqrt(size_map[k]/np.pi))
        for k, v in palette.items() if k != 'Background'
    ]
    ax.legend(handles=legend_elements, title="Node Type")

    # 축 라벨 및 제목 설정
    ax.set_title(f"3D UMAP Visualization of Traversal: '{_A.image_path.split('/')[-1]}'", fontsize=16, pad=20)
    ax.set_xlabel("UMAP Dimension 1")
    ax.set_ylabel("UMAP Dimension 2")
    ax.set_zlabel("UMAP Dimension 3")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize image-to-text traversal using 3D UMAP.")
    # 파일 경로 설정
    parser.add_argument("--image-path", default='./assets/green_dress.jpeg', help="Path to an image for performing traversal.")
    parser.add_argument("--text-embeddings-path", default='./embeddings/text_embeddings.json', help="Path to the pre-generated text embeddings JSON file.")
    parser.add_argument("--checkpoint-path", default='./checkpoints/meru_vit_b.pth', help="Path to the MERU model checkpoint.")
    parser.add_argument("--train-config", default='./configs/train_meru_vit_b.py', help="Path to the model's training config file.")
    
    # Traversal 옵션
    parser.add_argument("--steps", type=int, default=50, help="Number of interpolation steps for traversal.")
    
    # 시각화 옵션
    parser.add_argument("--fig-width", type=int, default=12, help="Width of the output plot figure.")
    parser.add_argument("--fig-height", type=int, default=12, help="Height of the output plot figure.")
    
    # UMAP 파라미터
    parser.add_argument("--n_neighbors", type=int, default=15, help="UMAP: Controls how UMAP balances local versus global structure in the data.")
    parser.add_argument("--min_dist", type=float, default=0.1, help="UMAP: Controls how tightly UMAP is allowed to pack points together.")
    parser.add_argument("--metric", type=str, default='cosine', help="UMAP: The metric to use for distance computation (e.g., 'cosine', 'euclidean').")

    _A = parser.parse_args()
    main(_A)