import tensorflow as tf
import  sys, os, json
from os.path import join, split
import numpy as np
import utils


utils    = utils.Utils()
DATADIR  = utils.DATADIR

class DataUtils(object):
    def __init__(self,args=None):
        self.DATASETDIR    = utils.DATASETDIR
        try:
            self.options = args.options
        except Exception:
            self.options = args

    def get_dataset(self,dataset_type,geolocation=None,encoded=False,flatten_features=True,rescale=True):
        features_encoded = None
        if dataset_type=="cifar10":
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
            features,labels = np.concatenate([x_train,x_test]), np.concatenate([y_train,y_test])
        elif dataset_type=="fashion_mnist":
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
            features,labels = np.concatenate([x_train,x_test]), np.concatenate([y_train,y_test])
        elif dataset_type=="cifar100":
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data(label_mode='coarse')
            features,labels = np.concatenate([x_train,x_test]), np.concatenate([y_train,y_test])
        else:
            print("Wrong dataset type, EXIT")
            sys.exit()
        if encoded:
            features_encoded = self.encode_format_dataset(features,dataset_type,flatten_features)
            self.feature_set,self.label_set = features_encoded, labels
        else:
            self.feature_set,self.label_set = features, labels
        if rescale:
            features = features/features.max()
        return features,labels, features_encoded


    def encode_format_dataset(self,features, dataset_type,flatten_features=True):
        # features, labels = self.get_dataset(dataset_type)
        print("init dataset_type",dataset_type)
        encoder = tf.keras.models.load_model(join(DATADIR,"encoders",dataset_type+".h5"))
        features = features/features.max()
        features_encoded = encoder.predict(features)
        features_encoded = np.reshape(features_encoded,[features_encoded.shape[0],
                                        np.prod(features_encoded.shape[1::])])
        return features_encoded






