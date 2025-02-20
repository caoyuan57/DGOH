from collections import namedtuple
import numpy as np
import torch
import scipy.io as sio
from util import *
import hdf5storage
from scipy.io import loadmat

_paths = {
    'Flickr': './dataset/MIRFLICKR.mat',
    'Flickr_vigb': './dataset/MIRFLICKR_vigb.mat',
    'COCO': './dataset/MS-COCO_vigb.mat',
    'word_vec_flickr': './dataset/word_vec_flickr.mat',
    'word_vec_coco':'./dataset/word_vec_coco.mat',
    'S_flickr': './dataset/S_flickr.mat'
}

dataind = namedtuple('dataind', ['idx_train', 'idx_val', 'idx_test', 'first', 'n_t'])


def normalize(x):
    l2_norm = np.linalg.norm(x, axis=1)[:, None]
    l2_norm[np.where(l2_norm == 0)] = 1e-6
    x = x / l2_norm
    return x


def zero_mean(x, mean_val=None):
    if mean_val is None:
        mean_val = np.mean(x, axis=0)
    x -= mean_val
    return x, mean_val


def load_data(dataset):
    if dataset == 'Flickr' or dataset == 'Flickr_vigb' or dataset == 'Flickr_vit' or dataset == 'MIRFLICKR_vigb_w2v':
        # The initial label features of flickr
        word_vec_data = sio.loadmat(_paths['word_vec_flickr'])
        word_vec_flickr = np.float32(word_vec_data['word_vec_flickr'])

        data = sio.loadmat(_paths[dataset])
        features = np.float32(data['XAll'])
        labels = np.int32(data['LAll'])

        idx_train = range(18000)
        idx_val = range(18000)
        idx_test = range(18000, 20015)
        cossim, valcossim = calcos(features, idx_train, idx_val, idx_test)

        features = normalize(features)
        features, mean_val = zero_mean(features)
        features = torch.tensor(features, dtype=torch.float32)
        labels = torch.tensor(labels)

        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)
        first = 4000
        n_t = 2000
        return features, labels, word_vec_flickr, cossim, valcossim, dataind(idx_train, idx_val, idx_test, first, n_t)
    elif 'COCO' in dataset:
        # The initial label features of coco
        word_vec_data = sio.loadmat(_paths['word_vec_coco'])
        word_vec_coco = np.float32(word_vec_data['word_vec_coco'])

        data = hdf5storage.loadmat(_paths[dataset])
        features = np.float32(data['data'])
        labels = np.int32(data['label'])

        idx_train = range(40000)
        idx_val = range(120218)
        idx_test = range(120218, 122218)

        cossim, valcossim = calcos(features, idx_train, idx_val, idx_test)

        features = normalize(features)
        features, mean_val = zero_mean(features)
        features = torch.tensor(features, dtype=torch.float32)

        labels = torch.tensor(labels)
        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)
        first = 4000
        n_t = 2000
        return features, labels, word_vec_coco, cossim, valcossim, dataind(idx_train, idx_val, idx_test, first, n_t)
    