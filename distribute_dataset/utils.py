import sys,os,json
from tensorflow.math import reduce_sum
from os.path import join,split
from pathlib import Path
import numpy as np



class Utils(object):
    def __init__(self):
        file_path           = os.path.realpath(__file__)
        self.HOMEDIR        = split(file_path)[0]
        self.DATADIR        = join(self.HOMEDIR ,'data')
        self.DATASETDIR     = join(self.DATADIR ,'training','distributed')

    def create_dir(self,dirpath):
        os.makedirs(dirpath, exist_ok=True)

    def split_noniid(self,train_idcs, train_labels, alpha, nclients):
        '''
        Splits a list of data indices with corresponding labels
        into subsets according to a dirichlet distribution with parameter
        alpha
        '''
        n_classes = int(train_labels.max()+1)
        label_distribution = np.random.dirichlet([alpha]*nclients, n_classes)
        class_idcs = [np.argwhere(train_labels[train_idcs]==y).flatten()
               for y in range(n_classes)]
        client_idcs = [[] for _ in range(nclients)]
        for c, fracs in zip(class_idcs, label_distribution):
            for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1]*len(c)).astype(int))):
                client_idcs[i] += [idcs]
        client_idcs = [train_idcs[np.concatenate(idcs)] for idcs in client_idcs]
        return client_idcs





if __name__ == '__main__':
    utils=Utils()
    print(utils.list_history_files("results","*.json"))
