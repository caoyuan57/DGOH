# DGOH
## This repository contains the matlab code for the paper: "Deep Graph Online Hashing for Image Retrieva" AAAI 2025.

# Parameter:
    dataset: Flickr_vigb for flickr  COCO for coco
    mode: supervised or unsupervised
    epochs: iteration times
    b_sz: batch size
    cuda: trainning device
    n_bits: Hash bits (16,32,48,...)
    hid_sz: the hidden layer of graphsage
    out_sz: the output layer of graphsage
    lab_num: the number of label categories
    alpha, beta,gama,delta,k,mju,lr: hyper-parameters

# dataCenter.py:
    Our experiment utilize two multi-label dataset, MIRFlickr and MS-COCO
    And we also have intial feature of labels, which initialized by word2vec.

# model.py:
    GraphSage model.

# util.py:
    The training process.

# main.py
    Set all parameters, then run.

# Whole process:
    1. Create your dataset in dataCenter.
    2. Initialize the label feature by word2vec.
    3. Set the above hyperparameter.
    4. run main.py
