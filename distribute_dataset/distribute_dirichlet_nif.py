import numpy as np
import os, sys, json
from os.path import join, split
from numpy.linalg import norm
import utils, datautils, argParser
from sklearn.cluster import KMeans, DBSCAN
import os
import seaborn as sns
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging

logging.getLogger('tensorflow').setLevel(logging.ERROR)

parser = argParser.argParser()
args = parser.args
print(args)
alpha = args.alpha
nclients = args.nclients
dataset_type = args.dataset_type
utils = utils.Utils()
ds = datautils.DataUtils(args)
HOMEDIR = utils.HOMEDIR
DATADIR = utils.DATADIR


def compute_ni(classes, features_encoded, labels, client_idcs, mu_arr):
    ni_per_class = []
    for class_idx in classes:
        print("class:", class_idx)
        iid = np.zeros(features_encoded[0].shape)
        for client, indices in enumerate(client_idcs):
            labels_sel = np.where(labels[indices] == class_idx)[0]
            if len(labels_sel) == 0: continue
            iid += features_encoded[labels_sel].mean(axis=0)
        ni_per_class.append(norm((mu_arr[class_idx] - iid / len(client_idcs))))
    return np.array(ni_per_class)


def main(alpha, nclients, dataset_type):
    OUTDIR = join(DATADIR, "training", "distributed", dataset_type)
    utils.create_dir(OUTDIR)
    PATH_TO_CLUSTER_INDICES = join(OUTDIR, "cluster_indices.npz")
    features, labels, features_encoded = ds.get_dataset(dataset_type, encoded=True, rescale=True)
    classes = np.arange(0, np.max(labels) + 1)

    """Generating tasks based on encoded feature non-iidness (K-means clustering)"""
    num_tasks = 20
    # kmeans = KMeans(n_clusters=num_tasks, random_state=0).fit(features_encoded)
    # coarse_label_counts = np.bincount(labels.flatten())
    #
    # # Plot histogram of coarse label counts
    # plt.bar(np.arange(20), coarse_label_counts)
    # coarse_label_counts = np.bincount(kmeans.labels_.flatten())
    #
    # # Plot histogram of coarse label counts
    # plt.bar(np.arange(20), coarse_label_counts, color="green", alpha=0.7)
    #
    # plt.xlabel('Coarse Label Group')
    # plt.ylabel('Sample Count')
    # plt.title('Histogram of CIFAR-100 Coarse Label Counts')
    # plt.show()

    """Generating tasks based on TSNE embeddings"""
    # X_embed = TSNE(n_components=20, perplexity=10, method='exact').fit_transform(features_encoded)
    # sns.scatterplot(
    #     x=X_embed[:, 0], y=X_embed[:, 1],
    #     hue=labels  # your labels (if you have any)
    # )
    # plt.show()

    """Generating tasks based on PCA"""

    # mu_arr,sigma_arr,label_distr = [],[],[]
    # for class_sel in classes:
    #     class_idc = np.where(labels==class_sel)[0]
    #     label_distr.append(len(class_idc))
    #     mu_arr.append(features_encoded[class_idc].mean(axis=0))
    #     sigma_arr.append(features_encoded[class_idc].var(axis=0))
    # ni_min_max=np.array([100,-1])
    # while True:
    #     emd_feature_hist,emd_label_hist = [],[]
    #     train_idcs  = np.random.permutation(features.shape[0])
    #     client_idcs = utils.split_noniid(train_idcs, labels[:,0], alpha=alpha, nclients=nclients)
    #     flag_continue=True
    #     client_idcs_sizes = np.zeros(nclients)
    #     for i,client_idc in enumerate(client_idcs):
    #         client_idcs_sizes[i]=len(client_idc)
    #         if len(client_idc)<128: ## discard low-sample client datasets
    #             flag_continue = False
    #             break
    #     if not flag_continue:continue
    #
    #     ni_per_class = compute_ni(classes,features_encoded,labels,client_idcs,mu_arr)
    #
    #     for _ni in ni_per_class: # discard empty client datasets
    #         if np.isnan(_ni): flag_continue = False;break
    #     if not flag_continue:continue
    #     ni_avg       = np.nanmean(ni_per_class)
    #     ni_min_max = np.array([min(ni_avg,ni_min_max[0]),max(ni_avg,ni_min_max[1])])
    #     print("Feature NonIID:",round(ni_avg,3) )
    #     meta = {"ni_feature":ni_avg,"ni_feature_per_class":list(ni_per_class.round(3))}
    #     ni_avg = round(ni_avg,2)
    #     subdir = join(OUTDIR,"N"+str(nclients)+"_alpha"+str(alpha)+"_nif"+str(ni_avg))
    #     utils.create_dir(subdir)
    #     with open(join(subdir,'meta.json'), 'w') as outfile:
    #         json.dump(meta, outfile)
    #     for client, indices in enumerate(client_idcs):
    #         filepath = join(OUTDIR,subdir,str(client)+".npz")
    #         np.savez(filepath,indices=indices)
    #     print("Saved to "+subdir,"\n")
    #     break


if __name__ == '__main__':
    main(alpha, nclients, dataset_type)
