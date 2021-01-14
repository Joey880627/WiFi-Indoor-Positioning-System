import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold, datasets
from dataset import WifiData
from config import Config
import torch
from torch.utils.data import DataLoader
from model import DNN_tsne
#Prepare the data
config = Config.from_json_file('config.json')
labeled_wifiData = WifiData(config, labeled = True)
X = []
y = []

# Load model
model = DNN_tsne().to(torch.device('cpu'))
model.load_state_dict(torch.load("./checkpoints/tsne/models/model.t7", map_location=torch.device('cpu')))
model.eval()
data_loader = DataLoader(labeled_wifiData, batch_size=64, shuffle=True)
for data, label in data_loader:
    logits, feat = model(data)
    X.extend(feat.detach().numpy())
    #X.extend(data.numpy())
    y.extend(label.numpy())
    # print(label)
X = np.array(X)[:5000]
y = np.array(y).astype(int)[:5000]

#t-SNE
X_tsne = manifold.TSNE(n_components=2, init='random', random_state=5, verbose=1).fit_transform(X)
print("Finish tSNE")
#Data Visualization
x_min, x_max = X_tsne.min(0), X_tsne.max(0)
X_norm = (X_tsne-x_min) / (x_max-x_min)  #Normalize
plt.figure(figsize=(8, 8))
print("Start plot")
#for i in range(X_norm.shape[0]):
#    plt.text(X_norm[i, 0], X_norm[i, 1], 'O', color=plt.cm.Set1(float(y[i, 0]+y[i, 1])/14), fontdict={'weight': 'bold', 'size': 9})
#plt.scatter(X_norm[:, 0], X_norm[:, 1], c=plt.cm.Set1(y[:, 0]*10+y[:, 1]))
plt.scatter(X_norm[:, 0], X_norm[:, 1], c=y[:, 1]/5, s=35, vmin=0, vmax=1)
plt.colorbar()
plt.xlim(0, 1)
plt.ylim(0, 1)

plt.xticks([])
plt.yticks([])
plt.show()