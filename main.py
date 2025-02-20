import torch
import argparse
import numpy as np
import time
from model import *
from util import *
from dataCenter import *

parser = argparse.ArgumentParser(description='Deep Graph online hashing')

parser.add_argument('--dataSet', type=str, default='Flickr_vigb')  # COCO Flickr_vigb
parser.add_argument('--label_net', type=bool, default=True)
parser.add_argument('--epochs', type=int, default=10)  # Flickr: 10
parser.add_argument('--b_sz', type=int, default=50)  # Flickr:50
parser.add_argument('--cuda', action='store_true', help='use CUDA')
parser.add_argument('--n_bits', type=int, default=32)
parser.add_argument('--hid_sz', type=int, default=128)  # Flickr or coco: 128
parser.add_argument('--out_sz', type=int, default=128)  # Flickr or coco: 128
parser.add_argument('--alpha', type=float, default=0.01)  # Flickr: 0.01
parser.add_argument('--beta', type=float, default=0.01)  # Flickr: 0.01
parser.add_argument('--mju', type=float, default=0.001)  # Flickr: 0.001 coco:0.0001
parser.add_argument('--delta', type=float, default=0.1)  # Flickr: 0.1
parser.add_argument('--gama', type=float, default=1)  # Flickr: 1
parser.add_argument('--k', type=float, default=1)  # Flickr: 1
parser.add_argument('--lr', type=float, default=0.001)  # Flickr:0.001 coco:0.001
args = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('DEVICE:', device)


"""
                                LOAD DATA FROM DATACENTER
=======================================================================================================================
"""
features, labels, word_vec, cossim, valcossim, dataind = load_data(args.dataSet)

start = time.perf_counter()

if args.dataSet == 'Flickr_vigb':
    lab_num = 24
elif args.dataSet == 'COCO':
    lab_num = 80

model = GraphSAGE(features.size(1), args.hid_sz, args.out_sz, labels.size(1), args.n_bits).to(device)
model_s = GraphSAGE(300, args.hid_sz, args.out_sz, labels.size(1), args.n_bits).to(device)

optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001)
optimizer_s = torch.optim.RMSprop(model_s.parameters(), lr=0.01)

model.train()
model_s.train()


"""
                                Training process
=======================================================================================================================
"""
for epoch in range(args.epochs):
    print('----------------------EPOCH %d-----------------------' % epoch)
    model, tH = apply_model(model, model_s, optimizer, optimizer_s, features, word_vec, lab_num, labels, args, dataind, device)
end = (time.perf_counter() - start)
print('trainTime:', end)


trn_binary, tst_binary, trn_label, tst_label = generate_code(model, cossim, valcossim, features, labels, tH, dataind, device)


"""
                                    MAP calculation
=======================================================================================================================
"""


if trn_label.ndim == 1:
    mAP = compute_mAP(trn_binary, tst_binary, trn_label, tst_label, args.n_bits)
else:
    mAP = CalcTopMap(trn_binary, tst_binary, trn_label, tst_label, args.n_bits)

