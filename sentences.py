from sentence_transformers import SentenceTransformer
import json
from tqdm import tqdm
from sklearn.cluster import KMeans 
import numpy as np
import math
import scipy as sp
import torch
import matplotlib.pyplot as plt
import faiss
from sklearn.manifold import TSNE
#from tsnecuda import TSNE
import random

color_list = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'lime', 'cyan', 'orange',
    'gray', 'purple', 'brown', 'pink', 'peru', 'gold', 'violet', 'navy', 'indigo', 'plum', 'deeppink']

REGION_GT = "/nfs_dongmin/VisualGenome/annotations/region_descriptions.json"
n_cluster = 1024
n_sen = -1
batch_size = 5000

print("Data loading...")
with open(REGION_GT) as f:
    region_gt = json.load(f)

sentences = []
embedding_sen = []

for i, regs in enumerate(tqdm(region_gt)):
    reg_gt = regs['regions']
    for j, reg in enumerate(reg_gt):
        sen = reg['phrase']
        sentences.append(sen)

print(f'number of sentences : {len(sentences)}')

print("Model loading...")
model = SentenceTransformer('paraphrase-MiniLM-L6-v2', device='cuda')
sentences = sentences[:n_sen]
#pool = model.start_multi_process_pool(target_devices=['cuda', 'cuda'])
embedding_sen = model.encode(sentences, batch_size=batch_size, show_progress_bar=True)
print(embedding_sen.shape)
#kmeans = KMeans(n_clusters = n_cluster)
kmeans = faiss.Kmeans(d=embedding_sen.shape[1], k=n_cluster, niter=40, verbose=True)
print("K-Means fitting")
#kmeans.fit(embedding_sen)
kmeans.train(embedding_sen)
#clustering = kmeans.labels_
dists, clustering = kmeans.index.search(embedding_sen, 1)
clustering = clustering.reshape(-1)
print(clustering)
idxs = np.where(clustering == 0)
idxs = np.array(idxs)
print(len(idxs))
print(len(sentences[idxs]))

"""print("T-SNE fitting")
tsne_np = TSNE(n_components = 2).fit_transform(embedding_sen)

for i, l in enumerate(range(n_cluster)):
    idxs = np.where(clustering == l)
    sc = tsne_np[idxs]
    plt.scatter(sc[:, 0], sc[:, 1], color = color_list[i], label=l, s=1)
#plt.legend(loc='best')
plt.savefig('/nfs_dongmin/prototype_densecap/prototype_densecap/results/result_sentence.png')"""