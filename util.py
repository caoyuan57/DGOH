import torch
import heapq
import math
import time
import numpy as np
from numpy.linalg import norm
import torch.nn as nn
from collections import namedtuple
from torch.autograd import Variable
from model import *


def calcos(features, idx_train, idx_val, idx_test):
    test = features[idx_test, :]
    val = features[len(idx_train):len(idx_val), :]
    train = features[:4000, :]
    norm1 = norm(test, axis=-1).reshape(test.shape[0], 1)
    norm2 = norm(train, axis=-1).reshape(1, train.shape[0])
    end_norm = np.dot(norm1, norm2)
    cossim = np.dot(test, train.T) / end_norm
    I = []
    k_cossim = np.zeros((test.shape[0], train.shape[0]), dtype=int)
    for i in range(len(cossim)):
        b = heapq.nlargest(200, range(len(cossim[i])), cossim[i].take)
        I.append(b)
    for i in range(k_cossim.shape[0]):
        k_cossim[i, I[i]] = 1

    norm1 = norm(val, axis=-1).reshape(val.shape[0], 1)
    end_norm = np.dot(norm1, norm2)
    valcossim = np.dot(val, train.T) / end_norm
    I = []
    k_valcossim = np.zeros((val.shape[0], train.shape[0]), dtype=int)
    for i in range(len(valcossim)):
        b = heapq.nlargest(200, range(len(valcossim[i])), valcossim[i].take)
        I.append(b)
    for i in range(k_valcossim.shape[0]):
        k_valcossim[i, I[i]] = 1
    return k_cossim, k_valcossim


def cross_entropy(logits, y):
    s = torch.exp(logits)
    logits = s / torch.sum(s, dim=1, keepdim=True)
    c = -(y * torch.log(logits)).sum(dim=-1)
    return torch.mean(c)


"""
The construction of our S,
def create_S(true_label, labels_1, labels_2):
    true_label: the label with weight
    labels_1,labels_2: label matrix
"""


def create_S(true_label, labels_1, labels_2, label_net):
    labels_1 = labels_1.detach().clone().cpu().numpy()
    labels_2 = labels_2.detach().clone().cpu().numpy()
    S = np.zeros((labels_1.shape[0], labels_2.shape[0]))
    if label_net:
        epsilon = 1e-10
        true_label = true_label[0]
        for i in range(labels_1.shape[0]):
            c = labels_1[i, :]
            for j in range(labels_2.shape[0]):
                d = labels_2[j, :]
                indices1 = np.where(c == 1)
                indices2 = np.where(d == 1)
                indices1_set = set(indices1[0])
                indices2_set = set(indices2[0])

                intersection = indices1_set & indices2_set
                array_intersection = list(intersection)
                e1 = sum(true_label[array_intersection])
                e2 = sum(true_label[array_intersection])
                m1 = np.sum(true_label[indices1])
                n1 = np.sum(true_label[indices2])
                S[i, j] = (e2 / (m1 + epsilon) + e1 / (n1 + epsilon)) / 2
    else:
        for i in range(labels_1.shape[0]):
            c = labels_1[i, :]
            for j in range(labels_2.shape[0]):
                d = labels_2[j, :]
                c = c.astype(int)
                d = d.astype(int)
                e = np.sum(c & d)
                m = np.sum(d)
                n = np.sum(c)
                S[i, j] = (e / m + e / n) / 2
    return S


def quantize(x):
    return torch.clamp((x - torch.min(x)) / (torch.max(x) - torch.min(x)), 0, 1)


def apply_model(model, model_s, optimizer, optimizer_s, features, word_vec, lab_num, labels, args, dataind, device):
    idx_train = dataind.idx_train
    trainsize = len(idx_train) + 1

    first = dataind.first
    n_t = dataind.n_t
    n_bits = args.n_bits
    b_sz = args.b_sz
    tH = []
    visited_nodes = set()

    fea_lab = torch.tensor(word_vec).to(device)
    fea_lab_clas_average = torch.zeros(1, lab_num)
    adj_lists_lab = [[], []]
    adj_matrix_lab_all = np.zeros((lab_num, lab_num))

    # Initialize the true_label
    avg_pool = nn.AvgPool1d(kernel_size=300)
    true_lab = avg_pool(torch.tensor(word_vec).to(device).unsqueeze(1)).squeeze(1)
    true_label = quantize(true_lab)
    true_label = true_label.clone().detach().cpu().numpy()
    true_label = true_label.reshape(1, lab_num)

    for t in range(first, trainsize, n_t):  # t-4000 6000 8000
        if t == first:
            train_node = idx_train[:first]
            batches = math.ceil(1 + (len(train_node) / (b_sz / 2)) - 2)
            for index in range(batches):
                if index == 0:
                    nodes_batch = train_node[index * b_sz:(index + 1) * b_sz]
                    visited_nodes |= set(nodes_batch)
                else:
                    nodes_batch = train_node[int(index * (b_sz / 2)): int(index * (b_sz / 2)) + b_sz]
                    visited_nodes |= set(nodes_batch[int(b_sz / 2):])

                if labels.ndim == 1:
                    labels_batch = labels[nodes_batch]
                else:
                    labels_batch = labels[nodes_batch, :]
                adj_lists = [[], []]
                Sim = create_S(true_label, labels_batch, labels_batch, args.label_net)
                for (row, col), val in np.ndenumerate(Sim):
                    if row != col and val >= 0.58:  # 0.58
                        # if row != col and val != 0:
                        adj_lists[0].append(row)
                        adj_lists[1].append(col)
                        adj_lists[0].append(col)
                        adj_lists[1].append(row)
                edges = torch.tensor(adj_lists, dtype=torch.long)
                Sim = torch.tensor(Sim).to(device)

                optimizer.zero_grad()

                logists, embs_batch, fea_hd, fea_convert, _ = model(
                    features[int(index * (b_sz / 2)): int(index * (b_sz / 2)) + b_sz].to(device), edges.to(device))
                if index % 2 == 0:
                    tH.append(embs_batch.data.cpu().numpy())

                quanloss = nn.MSELoss()

                if args.label_net:
                    lab_node = labels_batch[0, :].reshape(1, lab_num)
                    adj_matrix_lab = np.dot(np.transpose(lab_node), lab_node)
                    adj_matrix_lab_all = adj_matrix_lab_all + adj_matrix_lab
                    indices_get_fea = [[] for _ in range(b_sz)]
                    fea_lab_clas_batch = torch.zeros((b_sz, lab_num)).to(device)
                    for i in range(b_sz):
                        indices_get_fea[i].append(np.where(labels_batch[i, :] == 1)[0])

                    B = torch.sign(embs_batch)
                    binary_target = Variable(B).cuda()
                    labels_batch = labels_batch.to(torch.float)
                    labels_batch = labels_batch.to(device)

                    for (row, col), val in np.ndenumerate(adj_matrix_lab):
                        if row != col and val != 0 and adj_matrix_lab_all[row][col] == 0:  # >=0.6
                            adj_lists_lab[0].append(row)
                            adj_lists_lab[1].append(col)
                            adj_lists_lab[0].append(col)
                            adj_lists_lab[1].append(row)
                    edges_lab = torch.tensor(adj_lists_lab, dtype=torch.long)

                    optimizer_s.zero_grad()
                    clas_lab, fea_lab_32, fea_lab, useless, true_lab = model_s(fea_lab.to(device), edges_lab.to(
                        device))

                    for i in range(b_sz):
                        for item in indices_get_fea[i][0]:
                            fea_lab_clas_average = fea_lab_clas_average.clone().detach().to(device) + clas_lab[
                                item].clone().detach().to(device)

                        fea_lab_clas_average = fea_lab_clas_average / len(indices_get_fea)
                        fea_lab_clas_batch[i] += fea_lab_clas_average[0].to(device)

                    true_label = quantize(true_lab)
                    true_label = true_label.clone().detach().cpu().numpy()
                    true_label = true_label.reshape(1, lab_num)

                    loss_lab_clas = cross_entropy(fea_lab_clas_batch.to(device), labels_batch) * args.k
                else:
                    B = torch.sign(embs_batch)
                    binary_target = Variable(B).cuda()
                    labels_batch = labels_batch.to(torch.float)
                    labels_batch = labels_batch.to(device)
                    loss_lab_clas = 0

                loss = ((n_bits * Sim - embs_batch @ B.t()) ** 2).mean() * args.alpha  # Flickr: 0.01
                loss += quanloss(embs_batch, binary_target) * args.mju  # Flickr: 0.001
                loss += cross_entropy(logists, labels_batch) * args.delta
                loss += quanloss(fea_convert.to(device), labels_batch) * args.gama  # w*b - l
                loss_all = loss + loss_lab_clas
                print('Step [{}/{}], Loss: {:.4f}, Dealed Nodes [{}/{}] '.format(index + 1, batches, loss.item(),
                                                                                 len(visited_nodes), len(idx_train)))
                loss_all.backward()
                optimizer.step()
                optimizer_s.step()

        else:
            train_node = idx_train[t - n_t: t]
            label_node = labels[t - n_t: t]
            batches = math.ceil(len(train_node) / b_sz)

            for index in range(batches):
                nodes_batch = train_node[index * b_sz:(index + 1) * b_sz]
                visited_nodes |= set(nodes_batch)
                if labels.ndim == 1:
                    labels_batch = labels[nodes_batch]
                else:
                    labels_batch = labels[nodes_batch, :]
                adj_lists = [[], []]
                Sim_new = create_S(true_label, labels_batch, labels_batch, args.label_net)
                for (row, col), val in np.ndenumerate(Sim_new):
                    if row != col and val >= 0.2:
                        adj_lists[0].append(row)
                        adj_lists[1].append(col)
                        adj_lists[0].append(col)
                        adj_lists[1].append(row)
                edges = torch.tensor(adj_lists, dtype=torch.long)

                optimizer.zero_grad()
                logists, embs_batch, fea_lab, fea_convert, _ = model(
                    features[nodes_batch[0]: nodes_batch[-1] + 1, :].to(device), edges.to(device))
                tH.append(embs_batch.data.cpu().numpy())

                quanloss = nn.MSELoss()

                oH = torch.tensor(np.reshape(np.array(tH), (-1, n_bits))).to(device)

                B = torch.sign(embs_batch)
                binary_target = Variable(B).cuda()
                labels_batch = labels_batch.to(torch.float)
                labels_batch = labels_batch.to(device)

                # Select the old data
                sum_rows = torch.sum(labels_batch, dim=0)
                _, topk_indices = torch.topk(sum_rows, int(lab_num / 2))
                selected_tensor = labels[:4000, topk_indices]
                sum_tensor = torch.sum(selected_tensor, dim=1)
                sorted_tensor, indices = torch.sort(sum_tensor, descending=True)
                min_indices = indices[-100:]
                old_label = labels[min_indices, :]

                Sim_old = create_S(true_label, old_label, labels_batch, args.label_net)
                Sim_old = torch.tensor(Sim_old).to(device)

                Sim_new = torch.tensor(Sim_new).to(device)

                loss = ((n_bits * Sim_new - embs_batch @ B.t()) ** 2).mean() * args.alpha  # Flickr: 0.01
                loss += ((n_bits * Sim_old - oH[min_indices] @ embs_batch.t()) ** 2).mean() * args.beta  # Flickr: 0.01
                loss += quanloss(embs_batch, binary_target) * args.mju  # Flickr: 0.001
                loss += cross_entropy(logists, labels_batch) * args.delta
                loss += quanloss(fea_convert.to(device), labels_batch) * args.gama

                print('Step [{}/{}], Loss: {:.4f}, Dealed Nodes [{}/{}] '.format(index + 1, batches, loss.item(),
                                                                                 len(visited_nodes), len(idx_train)))
                loss.backward()
                optimizer.step()
    tH = np.concatenate(tH)
    return model, tH


def generate_code(model, cossim, valcossim, features, labels, tH, dataind, device):
    idx_train = dataind.idx_train
    idx_test = dataind.idx_test
    train_nodes = dataind.idx_val
    model.eval()
    trainemb = []
    for t in range(len(idx_train), len(train_nodes), 1000):
        train_node = train_nodes[t:t + 1000]
        tradj_list = [[], []]
        num_node = t - len(idx_train)
        affnty = valcossim[num_node: num_node + 1000, :]
        l = affnty.shape[0]
        r = affnty.shape[1]
        for (row, col), val in np.ndenumerate(affnty):
            if val != 0:
                tradj_list[0].append(row)
                tradj_list[1].append(col + l)
                tradj_list[0].append(col + l)
                tradj_list[1].append(row)
        edges = torch.tensor(tradj_list, dtype=torch.long)
        fea = torch.cat((features[train_node[0]: train_node[-1] + 1, :], features[:r, :]))
        logists, tremb, useless, useless_, _ = model(fea.to(device), edges.to(device))
        trainemb.append(tremb[:len(train_node), :].data.cpu().numpy())
    if len(trainemb):
        trainemb = np.concatenate(trainemb)
        trainembs = np.vstack((tH, trainemb))
    else:
        trainembs = tH
    trainembs = torch.Tensor(trainembs)
    # fea = torch.cat((features[59000:60000, :],features[:1000, :]))

    tsadj_list = [[], []]
    affnty = cossim[:, :]
    l = affnty.shape[0]
    r = affnty.shape[1]
    for (row, col), val in np.ndenumerate(affnty):
        if val != 0:
            tsadj_list[0].append(row)
            tsadj_list[1].append(col + l)
            tsadj_list[0].append(col + l)
            tsadj_list[1].append(row)
    tsadj_list = torch.tensor(tsadj_list, dtype=torch.long)
    fea = torch.cat((features[idx_test, :], features[:r, :]))
    logists, testembs, useless, useless_, _ = model(fea.to(device), tsadj_list.to(device))

    tst_label = labels[idx_test]
    trn_label = labels[train_nodes]
    tst_binary = torch.sign(testembs[:len(idx_test), :]).to(device)
    trn_binary = torch.sign(trainembs).to(device)

    return trn_binary, tst_binary, trn_label, tst_label


def compute_mAP(trn_binary, tst_binary, trn_label, tst_label, n_bits):
    AP = []
    Ns = torch.arange(1, trn_binary.size(0) + 1)
    for i in range(tst_binary.size(0)):
        query_label, query_binary = tst_label[i], tst_binary[i]
        _, query_result = torch.sum(
            (query_binary != trn_binary).long(), dim=1).sort()

        correct = torch.as_tensor((query_label == trn_label[query_result])).float()
        P = torch.cumsum(correct, dim=0) / Ns
        AP.append(torch.sum(P * correct) / torch.sum(correct))
    mAP = torch.mean(torch.Tensor(AP))
    print('%d bits: mAP: %.4f' % (n_bits, mAP))
    return mAP


def CalcHammingDist(B1, B2):
    q = B2.shape[1]
    if len(B1.shape) < 2:
        B1 = B1.unsqueeze(0)
    distH = 0.5 * (q - B1.mm(B2.transpose(0, 1)))
    return distH


def CalcTopMap(rB, qB, retrieval_L, query_L, n_bits, k=None):
    # qB: {-1,+1}^{mxq}
    # rB: {-1,+1}^{nxq}
    # query_L: {0,1}^{mxl}
    # retrieval_L: {0,1}^{nxl}
    start = time.perf_counter()
    num_query = query_L.shape[0]
    map = 0
    if k is None:
        k = retrieval_L.shape[0]
    for iter in range(num_query):
        q_L = query_L[iter]
        if len(q_L.shape) < 2:
            q_L = q_L.unsqueeze(0)  # [1, hash length]
        gnd = (q_L.mm(retrieval_L.transpose(0, 1)) > 0).squeeze().type(torch.float32)
        tsum = torch.sum(gnd)
        if tsum == 0:
            continue
        hamm = CalcHammingDist(qB[iter, :], rB)
        _, ind = torch.sort(hamm)
        end = (time.perf_counter() - start)
        # print("QueryTime:", end)
        ind.squeeze_()
        gnd = gnd[ind]
        total = min(k, int(tsum))
        count = torch.arange(1, total + 1).type(torch.float32)
        tindex = torch.nonzero(gnd)[:total].squeeze().type(torch.float32) + 1.0
        if tindex.is_cuda:
            count = count.cuda()
        map = map + torch.mean(count / tindex)
    mAP = map / num_query
    print('%d bits: mAP: %.4f' % (n_bits, mAP))
    return map
