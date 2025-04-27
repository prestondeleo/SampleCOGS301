import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

def detect(data:list)->None:
    message_lengths = np.array([len(s) for s in data]).reshape(-1, 1)
    neighbors = NearestNeighbors(n_neighbors=3)
    neighbors.fit(message_lengths)
    distances, _ = neighbors.kneighbors(message_lengths)
    mean_distances = distances[:, 1:].mean(axis=1)
    threshold = np.percentile(mean_distances, 95)
    anom_sarcasm = mean_distances > threshold
    anom_messages = [data[i] for i in range(len(data)) if anom_sarcasm[i]]
    print(anom_messages)
    plt.scatter(range(len(message_lengths)), message_lengths)
    plt.scatter(np.where(anom_sarcasm)[0], message_lengths[anom_sarcasm], c='purple', label='novel examples')
    plt.xlabel('Index')
    plt.ylabel('Length of Sarcastic Message')
    plt.title('Novel Sarcasm')
    plt.legend()
    plt.show()