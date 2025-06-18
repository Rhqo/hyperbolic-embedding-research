import json
import argparse
import numpy as np
import torch
from PIL import Image

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 3D 플롯을 위한 임포트
from sklearn.decomposition import PCA
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
    # --- 1, 2, 3 단계는 이전과 동일 ---
    print("Step 1-3: Loading data, model and finding traversal path...")
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

    # --- 4. PCA를 위한 데이터 준비 및 3D 차원 축소 ---
    print("\nStep 4: Performing 3D PCA for visualization...")
    all_embeddings = torch.cat([image_feats.unsqueeze(0), text_feats_pool]).cpu().numpy()
    all_labels = ["[IMAGE]"] + text_pool
    all_types = ['Image'] + ['Background'] * (len(text_pool) - 1) + ['Root']
    for i, label in enumerate(all_labels):
        if label in traversal_texts and label not in ["[IMAGE]", "[ROOT]"]:
            all_types[i] = 'Traversal Node'

    # PCA 컴포넌트 수를 3으로 변경
    pca = PCA(n_components=3)
    embeddings_3d = pca.fit_transform(all_embeddings)

    # --- 5. 3D 시각화 ---
    print("Step 5: Generating 3D plot...")
    plt.style.use('default')
    fig = plt.figure(figsize=(_A.fig_width, _A.fig_height), dpi=120)
    
    # 3D 축 생성
    ax = fig.add_subplot(111, projection='3d')
    
    palette = {'Image': 'red', 'Traversal Node': 'darkorange', 'Root': 'black', 'Background': 'lightgray'}
    sizes = {'Image': 200, 'Traversal Node': 100, 'Root': 200, 'Background': 20}
    
    # 타입에 따라 색상과 크기 리스트 생성
    color_map = [palette[t] for t in all_types]
    size_map = [sizes[t] for t in all_types]

    # 3D 산점도 그리기
    ax.scatter(
        embeddings_3d[:, 0], embeddings_3d[:, 1], embeddings_3d[:, 2],
        c=color_map,
        s=size_map,
        alpha=0.7,
        depthshade=True # 입체감을 위해 깊이에 따라 음영 조절
    )
    
    # Traversal 경로를 3D로 그리기
    path_indices = [all_labels.index(label) for label in traversal_texts]
    path_coords = embeddings_3d[path_indices]
    ax.plot(
        path_coords[:, 0], path_coords[:, 1], path_coords[:, 2],
        color='gray', linestyle='--', linewidth=2, marker='o', markersize=0, zorder=10
    )

    # 경로 노드에 텍스트 라벨 추가
    for i, label in enumerate(all_labels):
        if all_types[i] != 'Background':
            ax.text(
                embeddings_3d[i, 0], embeddings_3d[i, 1], embeddings_3d[i, 2],
                f" {label}",  # 텍스트가 점에 겹치지 않도록 공백 추가
                fontsize=9,
                fontweight='bold',
                color=palette.get(all_types[i], 'black')
            )

    # 제목 및 축 라벨 설정 (z축 추가)
    ax.set_title(f"3D PCA Visualization of Traversal: '{_A.image_path.split('/')[-1]}'", fontsize=16, pad=20)
    ax.set_xlabel("Principal Component 1", fontsize=12)
    ax.set_ylabel("Principal Component 2", fontsize=12)
    ax.set_zlabel("Principal Component 3", fontsize=12)

    # 범례(Legend) 수동 생성
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=key,
                                  markerfacecolor=val, markersize=np.sqrt(sizes[key])/2)
                       for key, val in palette.items()]
    ax.legend(handles=legend_elements, title="Node Type")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize image-to-text traversal using 3D PCA.")
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

    _A = parser.parse_args()
    main(_A)