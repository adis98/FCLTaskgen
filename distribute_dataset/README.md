
# Dataset distribution

Distributes dataset among N clients following a Dirichlet distribution with parameter alpha and computes
the skew index as the deviation fromn the center of mass in feature space. The latter operation
requires the features to be dimensionally downsized for computational efficiency and
salient feature extraction. We trained 3 different encoders in data/encoders but any feature extractor layer
from Imagenet or VGG could have done the job.

# Pre-requisites
- Tensorflow
- Numpy

# Usage

´python3 distribute_dirichlet_nif.py --alpha ALPHA --nclient NCLIENTS --dataset_type DATASET´

- Example

´python3 distribute_dirichlet_nif.py --alpha 1.0 --nclient 10 --dataset_type cifar10´

- Datasets are saved in /data/distributed folder. Only indices are saved.
