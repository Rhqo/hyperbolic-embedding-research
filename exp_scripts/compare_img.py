import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend for saving plots

dress_0 = 2
dress_1 = 3


with open(f'./embeddings/hyperbolic_embeddings.json', 'r') as file:
    data = json.load(file)
array_data = np.array([v[0] for v in data.values()], dtype=float)
data_0 = array_data[dress_0]
data_1 = array_data[dress_1]

# Plot the embeddings
plt.figure(figsize=(10, 5))
plt.plot(data_0, label=f'dress_0', color = 'red', alpha=1)
plt.plot(data_1, label=f'dress_1', color = 'blue', alpha=0.5)
plt.title('Dimension-wise Embeddings Line Chart')
plt.xlabel('Dimension Index')
plt.ylabel('Value')
plt.legend()
plt.savefig('compare_img_json_chart.png')
plt.close()