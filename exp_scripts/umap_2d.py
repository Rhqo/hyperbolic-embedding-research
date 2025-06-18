import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import umap.umap_ as umap # UMAP import

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

# UMAP으로 변경 (n_components=2, random_state=42)
reducer = umap.UMAP(n_components=2, random_state=42)
all_embeddings_umap = reducer.fit_transform(all_embeddings_scaled)


image_embeddings_umap = all_embeddings_umap[:len(image_embeddings)]
text_embedding_umap = all_embeddings_umap[len(image_embeddings)]
root_embedding_umap = all_embeddings_umap[-1]


plt.figure(figsize=(10, 8))

plt.scatter(image_embeddings_umap[:, 0], image_embeddings_umap[:, 1], c='red', alpha=0.7, marker='o', label='Image Embeddings')
plt.scatter(text_embedding_umap[0], text_embedding_umap[1], c='blue', marker='x', s=150, label='A photo of a dress')
plt.scatter(root_embedding_umap[0], root_embedding_umap[1], c='green', marker='^', s=150, label='Root Embedding')

plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.title('2D UMAP of Embeddings')
plt.grid(True)
plt.legend()
plt.show()