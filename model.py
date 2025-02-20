import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import SAGEConv


class GraphSAGE(torch.nn.Module):
    def __init__(self, feature, hidden, out_size, num_classes, n_bits):
        super(GraphSAGE, self).__init__()
        self.sage1 = SAGEConv(feature, hidden)
        self.sage2 = SAGEConv(hidden, out_size)
        self.conv = nn.Linear(out_size, n_bits)
        self.clas = nn.Linear(out_size, num_classes)
        self.hd = nn.Linear(out_size, 300)
        self.BN = nn.BatchNorm1d(n_bits)
        self.tl = nn.Linear(n_bits, 1)
        self.act = nn.Tanh()
        self.convert = nn.Linear(n_bits, num_classes)

    def forward(self, features, edges):
        features = self.sage1(features, edges)
        features = F.relu(features)
        features = F.dropout(features, training=self.training)
        features = self.sage2(features, edges)
        features = F.relu(features)
        features = F.dropout(features, training=self.training)  # 增加了dropout
        fea_lab = self.hd(features)
        logists = self.clas(features)
        pre_hidden_embs = self.conv(features)
        pre_hidden_embs = self.BN(pre_hidden_embs)
        true_lab = self.tl(pre_hidden_embs)
        out = self.act(pre_hidden_embs)
        fea_convert = self.convert(out)

        return logists, out, fea_lab, fea_convert, true_lab
