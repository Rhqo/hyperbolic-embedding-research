import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
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

pca = PCA(n_components=3)
all_embeddings_3d = pca.fit_transform(all_embeddings_scaled)

explained_variance_ratio = pca.explained_variance_ratio_
print(f"PCA 설명된 분산 비율: {explained_variance_ratio}")

image_embeddings = all_embeddings_3d[:len(image_embeddings)]
text_embedding = all_embeddings_3d[len(image_embeddings)]
root_embedding = all_embeddings_3d[-1]

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(image_embeddings[:, 0], image_embeddings[:, 1], image_embeddings[:, 2], c='red', alpha=0.7, marker='o', label='image embeddings')
ax.scatter(text_embedding[0], text_embedding[1], text_embedding[2], c='blue', marker='^', s=150, label='a photo of a dress')
ax.scatter(root_embedding[0], root_embedding[1], root_embedding[2], c='black', marker='x', s=150, label='root embedding')
ax.set_xlabel('PCA1')
ax.set_ylabel('PCA2')
ax.set_zlabel('PCA3')
plt.legend()
plt.show()