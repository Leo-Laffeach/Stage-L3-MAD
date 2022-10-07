import numpy.random as npr
import torch
import torch.nn as nn
import numpy.random as npr
import ot
import tslearn.metrics as tslm
from torch.utils.data import Dataset
import numpy as np
import warnings
from MADCNN import CNNMAD


"""series0_source1 = torch.zeros(size=(1000, 200, 1)).type(torch.float)
series0_source0 = torch.ones(size=(1000, 80, 1)).type(torch.float)
series0_source = torch.cat((series0_source0, series0_source1), dim=1)
series1_source1 = torch.zeros(size=(1000, 80, 1)).type(torch.float)
series1_source0 = torch.ones(size=(1000, 200, 1)).type(torch.float)
series1_source = torch.cat((series1_source0, series1_source1), dim=1)
series_source = torch.cat((series0_source, series1_source), dim=0)
series0_target0 = torch.zeros(size=(1500, 400, 1)).type(torch.float)
series0_target1 = torch.ones(size=(1500, 60, 1)).type(torch.float)
series0_target = torch.cat((series0_target0, series0_target1), dim=1)
series1_target0 = torch.zeros(size=(1500, 60, 1)).type(torch.float)
series1_target1 = torch.ones(size=(1500, 400, 1)).type(torch.float)
series1_target = torch.cat((series1_target0, series1_target1), dim=1)
series_target = torch.cat((series0_target, series1_target), dim=0)"""
source_size = 20
target_size = 30
series_source = torch.rand(size=(source_size, 100, 1))
series_target = torch.rand(size=(target_size, 110, 1))

labels0 = torch.zeros(size=(int(source_size/2),)).type(torch.long)
labels1 = torch.ones(size=(int(source_size/2),)).type(torch.long)
labels = torch.cat((labels0, labels1))

labels0_target = torch.zeros(size=(int(target_size/2),)).type(torch.long)
labels1_target = torch.ones(size=(int(target_size/2),)).type(torch.long)
labels_target = torch.cat((labels0_target, labels1_target))

cnnmad = CNNMAD(name="", batchsize=10000, beta=1.0, alpha=1.0, channel=1, valid_target_label=labels_target,
                train_target_data=series_target, test_target_data=series_target, test_target_label=labels_target,
                test_source_label=labels, test_source_data=series_source, train_source_data=series_source,
                train_source_label=labels, valid_source_data=series_source, valid_target_data=series_target,
                num_class=2, valid_source_label=labels)

cnnmad.set_current_batchsize(source_size)
cnnmad.set_current_batchsize(target_size, train=False)
out_source, conv_source = cnnmad.forward(cnnmad.trainSourceData.transpose(1, 2))
out_target, conv_target = cnnmad.forward(cnnmad.trainTargetData.transpose(1, 2), train=False)
cnnmad.mad(out_conv_source=conv_source, out_conv_target=conv_target, labels=labels)
cnnmad.loss_CNN_MAD(labels, conv_source, conv_target, out_target)
DTW = np.double(cnnmad.DTW)
OT = np.double(cnnmad._OT)
DTWf = cnnmad.DTW
OTf = cnnmad._OT.astype(np.float32)
out_conv_source = conv_source.type(torch.long)
out_conv_target = conv_target.type(torch.long)
#  Torch version
source_sq = out_conv_source ** 2
target_sq = out_conv_target ** 2
res = torch.empty(size=(source_size, target_size)).type(torch.long)
for cl in range(0, 2):
    idx_cl = torch.where(labels == cl)
    pi_DTW = DTW[cl]
    pi_DTW = torch.tensor(pi_DTW).long()
    C1 = torch.matmul(source_sq[idx_cl], torch.sum(pi_DTW, dim=1)).sum(-1)
    C2 = torch.matmul(target_sq, torch.sum(pi_DTW.T, dim=1)).sum(-1)
    C3 = torch.tensordot(torch.matmul(out_conv_source[idx_cl], pi_DTW), out_conv_target, dims=([1, 2], [1, 2]))
    res[idx_cl] = C1[:, None] + C2[None, :] - 2 * C3
out_CNN = torch.tensor(OT) * res

# Numpy version
out_np_source = out_conv_source.detach().numpy()
out_np_target = out_conv_target.detach().numpy()
np_source_sq = out_np_source ** 2
np_target_sq = out_np_target ** 2
res_np = np.double(np.empty(shape=(source_size, target_size)))
for cl in range(0, 2):
    pi_DTW = DTW[cl]
    idx_cl = np.where(labels == cl)
    C1 = np.dot(np_source_sq[idx_cl], np.sum(pi_DTW, axis=1)).sum(-1)
    C2 = np.dot(np_target_sq, np.sum(pi_DTW.T, axis=1)).sum(-1)
    C3 = np.tensordot(np.dot(out_np_source[idx_cl], pi_DTW), out_np_target.transpose(0, -1, 1), axes=([1, 2], [2, 1]))
    res_np[idx_cl] = C1[:, None] + C2[None, :] - 2 * C3
np_out_CNN = OT * res_np

# Numpy version float
out_np_source = out_conv_source.float().detach().numpy()
out_np_target = out_conv_target.float().detach().numpy()
np_source_sq = out_np_source ** 2
np_target_sq = out_np_target ** 2
res_np_float = np.empty(shape=(source_size, target_size)).astype(np.float32)
print(type(res_np_float[0, 0]), type(OTf[0, 0]))
print(type(out_np_source[0, 0, 0]))
for cl in range(0, 2):
    pi_DTW = DTWf[cl].astype(np.float32)
    idx_cl = np.where(labels == cl)
    C1 = np.dot(np_source_sq[idx_cl], np.sum(pi_DTW, axis=1)).sum(-1)
    C2 = np.dot(np_target_sq, np.sum(pi_DTW.T, axis=1)).sum(-1)
    C3 = np.tensordot(np.dot(out_np_source[idx_cl], pi_DTW), out_np_target.transpose(0, -1, 1), axes=([1, 2], [2, 1]))
    res_np_float[idx_cl] = C1[:, None] + C2[None, :] - 2 * C3
print(type(res_np_float[0, 0]), type(OTf[0, 0]))
np_out_CNN_float = OTf * res_np_float

#  Torch version float
source_sq = conv_source ** 2
target_sq = conv_target ** 2
res_float = torch.empty(size=(source_size, target_size))
for cl in range(0, 2):
    idx_cl = torch.where(labels == cl)
    pi_DTW = DTWf[cl]
    pi_DTW = torch.tensor(pi_DTW).type(torch.float)
    C1 = torch.matmul(source_sq[idx_cl], torch.sum(pi_DTW, dim=1)).sum(-1)
    C2 = torch.matmul(target_sq, torch.sum(pi_DTW.T, dim=1)).sum(-1)
    C3 = torch.tensordot(torch.matmul(conv_source[idx_cl], pi_DTW), conv_target, dims=([1, 2], [1, 2]))
    res_float[idx_cl] = C1[:, None] + C2[None, :] - 2 * C3
out_CNN_float = torch.tensor(OTf) * res_float

print('MAD score with numpy float', type(np_out_CNN_float.sum()), np_out_CNN_float.sum())
print("MAD score with numpy double ", np_out_CNN.sum())
print("MAD score with torch double ", out_CNN.sum().item())
print("MAD score with torch float", out_CNN_float.sum().item())
