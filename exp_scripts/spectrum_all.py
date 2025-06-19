import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
import os
import sys
import matplotlib
if sys.platform != 'darwin':
    matplotlib.use('TkAgg')

from encoders import Encoder


# --- 설정 ---
# 모델 설정 파일 및 체크포인트 경로
TRAIN_CONFIG_PATH = './configs/train_meru_vit_b.py'
CHECKPOINT_PATH = './checkpoints/meru_vit_b.pth'

# 시각화할 이미지 경로와 텍스트 정의
image_paths = [
    './datasets/fashion200k/casual_and_day_dresses/51727804/51727804_0.jpeg',
    './datasets/fashion200k/casual_and_day_dresses/54686996/54686996_0.jpeg'
]
texts = [
    'a photo of a dress',
    'a photo of a dog',
    'asdfasdflkjhlkj'
]

encoder = Encoder(train_config_path=TRAIN_CONFIG_PATH, checkpoint_path=CHECKPOINT_PATH)


all_embeddings_list = []
ytick_labels = []

# 이미지 임베딩 생성
for path in image_paths:
    image_embedding = encoder.image_encoder(path)
    all_embeddings_list.append(image_embedding)
    ytick_labels.append(os.path.basename(path)) # 레이블로 파일명 사용

# 텍스트 임베딩 생성
for text in texts:
    text_embedding = encoder.text_encoder(text)
    all_embeddings_list.append(text_embedding)
    ytick_labels.append(f'"{text}"') # 레이블로 텍스트 사용

# --- 3. 데이터 준비 ---
# 모든 임베딩을 하나의 큰 NumPy 배열로 쌓습니다.
# 각 임베딩이 (1, 512) 형태라고 가정하면, final_data는 (N, 512) 형태가 됩니다.
# 여기서 N은 총 이미지와 텍스트의 개수입니다.
final_data = np.vstack(all_embeddings_list)

# --- 4. 히트맵 시각화 ---
# 전체 데이터의 최소/최대값을 기준으로 색상 범위를 정합니다.
# 이렇게 하면 색상 스케일이 데이터에 최적화됩니다.
vmin = final_data.min()
vmax = final_data.max()

plt.figure(figsize=(20, 10)) # figsize는 데이터 개수에 맞게 조절
plt.imshow(
    final_data,
    aspect='auto',
    cmap='coolwarm',
    norm=matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
)

plt.colorbar(label='Embedding Value')
plt.title(f'Image and Text Embeddings Heatmap ({final_data.shape[0]} sources, {final_data.shape[1]} dims)')

# 각 임베딩 사이에 구분선 추가
# i + 0.5는 i번째 행과 i+1번째 행의 경계를 의미합니다.
for i in range(final_data.shape[0] - 1):
    plt.axhline(i + 0.5, color='black', linestyle='--', linewidth=1)

# y축 눈금 및 레이블 설정
# 각 행의 중앙에 눈금을 위치시킵니다.
plt.yticks(ticks=np.arange(len(ytick_labels)), labels=ytick_labels)

plt.xlabel('Dimension Index')
plt.ylabel('Data Source')

# 레이블이 잘리지 않도록 레이아웃을 조정합니다.
plt.tight_layout()

# --- 6. 파일로 저장 ---
output_filename = 'embeddings_heatmap.png'
plt.savefig(output_filename)
plt.show()
plt.close()
