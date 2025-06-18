import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from encoders import Encoder

TRAIN_CONFIG_PATH = './configs/train_meru_vit_b.py'
CHECKPOINT_PATH = './checkpoints/meru_vit_b.pth'

encoder = Encoder(train_config_path=TRAIN_CONFIG_PATH, checkpoint_path=CHECKPOINT_PATH)

def load_embeddings(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    embeddings = [embedding for _, embedding in data.items()]
    embeddings = np.array(embeddings)
    if len(embeddings.shape) > 2:
        embeddings = embeddings.reshape(embeddings.shape[0], -1)
    return embeddings


image_embeddings = load_embeddings('./embeddings/hyperbolic_embeddings.json')
text_embedding = encoder.text_encoder('a photo of a dress').flatten().reshape(1, -1)
root_embedding = np.zeros((1, image_embeddings.shape[1]))

all_embeddings = np.vstack([image_embeddings, text_embedding, root_embedding])

scaler = StandardScaler()
all_embeddings_scaled = scaler.fit_transform(all_embeddings)

pca = PCA(n_components=2)
all_embeddings_2d = pca.fit_transform(all_embeddings_scaled)

explained_variance_ratio = pca.explained_variance_ratio_
print(f"PCA 설명된 분산 비율: {explained_variance_ratio}")

image_embeddings_2d = all_embeddings_2d[:len(image_embeddings)]
text_embedding_2d = all_embeddings_2d[len(image_embeddings)]
root_embedding_2d = all_embeddings_2d[-1]

plt.figure(figsize=(10, 8))

plt.scatter(image_embeddings_2d[:, 0], image_embeddings_2d[:, 1], c='red', alpha=0.7, marker='o', label='Image Embeddings')
plt.scatter(text_embedding_2d[0], text_embedding_2d[1], c='blue', marker='^', s=150, label='A photo of a dress')
plt.scatter(root_embedding_2d[0], root_embedding_2d[1], c='black', marker='x', s=150, label='Root Embedding')

plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.title('2D PCA of Embeddings')
plt.grid(True)
plt.legend()
plt.show()