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

TRAIN_CONFIG_PATH = './configs/train_meru_vit_b.py'
CHECKPOINT_PATH = './checkpoints/meru_vit_b.pth'

encoder = Encoder(train_config_path=TRAIN_CONFIG_PATH, checkpoint_path=CHECKPOINT_PATH)

# Define image paths
image_paths = ['./datasets/fashion200k/casual_and_day_dresses/51727804/51727804_0.jpeg', './datasets/fashion200k/casual_and_day_dresses/54686996/54686996_0.jpeg']
texts = ['a photo of a dress', 'a photo of a dog']

image_0 = encoder.image_encoder(image_paths[0]).flatten()
image_1 = encoder.image_encoder(image_paths[1]).flatten()
text_0 = encoder.text_encoder(texts[0]).flatten()
text_1 = encoder.text_encoder(texts[1]).flatten()

# Plot the embeddings
plt.figure(figsize=(10, 5))
plt.plot(image_0, label=os.path.basename(image_paths[0]), color='red', alpha=1)
plt.plot(image_1, label=os.path.basename(image_paths[1]), color='green', alpha=0.5)
plt.plot(text_0, label=f'\"{texts[0]}\"', color='blue', alpha=0.5)
plt.plot(text_1, label=f'\"{texts[1]}\"', color='orange', alpha=0.5)
plt.title('Dimension-wise Embeddings Line Chart')
plt.xlabel('Dimension Index')
plt.ylabel('Value')
plt.legend()
plt.show()
# plt.savefig('compare_img_chart.png')
plt.close()