import json
with open("X2D_output.json", "r", encoding='utf-8') as json_file:
    data_memory = json.load(json_file)

with open("y.json", "r", encoding='utf-8') as json_file:
    y = json.load(json_file)
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.font_manager as font_manager
#font_manager.fontManager.addfont('path/to/Times New Roman.ttf') # Specify the path to the TTF file of the font if not installed system-wide.
plt.rcParams['font.family'] = 'Times New Roman'

y_over = list()
empty = list()
for k in range(len(y)):
    item = y[k]
    if item >= 2.5:
        empty.append(k)

y = np.array(y)

X_2d = np.array(data_memory)
X_over = np.array([X_2d[e] for e in empty])


plt.figure(figsize=(8, 6))

print(X_2d.shape, y.shape)
print(X_over.shape)
scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap='rainbow', s=2)
#scatter2 = plt.scatter(X_over[:, 0], X_over[:, 1], c='red', s = 2)

cbar = plt.colorbar(scatter)
cbar.set_label('estimation of action state value', rotation=270, labelpad=15)

plt.xlabel('Feature1')
plt.ylabel('Feature2')
#plt.title('2D feature map by t-SNE dimension reduction and value estimation ')
plt.savefig('policy.png', dpi = 1200)
plt.show()