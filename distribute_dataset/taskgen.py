import datautils, utils
import numpy as np
from matplotlib import pyplot as plt
import os
import random
from scipy.spatial.distance import pdist

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

dataset_type = 'cifar100'
nclients = 4
alphas = [1, 10, 100, 1000]
min_samples_per_client = 128
ds = datautils.DataUtils()
ut = utils.Utils()
seedval = 41


def compute_fni(feature_means):
    nparr = np.array(client_mean_features)
    distances = pdist(nparr)
    distances_rms = distances/np.sqrt(feature_means[0].shape[0])
    return np.mean(distances_rms)


if __name__ == "__main__":
    random.seed(seedval)
    np.random.seed(seedval)
    features, labels, features_encoded = ds.get_dataset(dataset_type, encoded=True, rescale=True)
    features_norm = (features_encoded - np.min(features_encoded)) / (
            np.max(features_encoded) - np.min(features_encoded))

    classes = np.arange(0, np.max(labels) + 1)
    train_idcs = client_idcs = None
    y = []
    for alpha in alphas:
        n_trials = 10
        feature_nis = []
        for i in range(n_trials):
            client_mean_features = []
            while True:
                train_idcs = np.random.permutation(features.shape[0])
                client_idcs = ut.split_noniid(train_idcs, labels[:, 0], alpha=alpha, nclients=nclients)
                if min([len(i) for i in client_idcs]) >= min_samples_per_client:
                    break
                else:
                    continue
            for cli in range(nclients):
                feats = features_norm[client_idcs[cli]]

                client_mean_features.append(np.mean(feats, axis=0))

            feature_nis.append(compute_fni(client_mean_features))
        y.append(feature_nis)
    x = np.arange(len(alphas))
    for xe, ye in zip(x, y):
        plt.scatter([xe] * len(ye), ye)

    plt.xticks(x, alphas)
    plt.show()



