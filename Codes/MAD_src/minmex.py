import numpy.random as npr
import torch
import torch.nn as nn
import numpy.random as npr
import ot
import tslearn.metrics as tslm
from torch.utils.data import Dataset
import numpy as np
import warnings

source = torch.rand(size=(200, 1, 200))
target = torch.rand(size=(200, 1, 200))
cost_OT = npr.rand(200, 200)
OT = np.double(ot.emd(np.ones(200) / 200, np.ones(200) / 200, cost_OT))
DTW = np.eye(200, dtype=np.double)


#  Torch version
source_sq = source ** 2
print("torch sum", source.sum().item())
print('source sum', source_sq.sum().item())
target_sq = target ** 2
res = torch.zeros(size=(200, 200))
out_CNN_before = torch.tensor(OT) * res

pi_DTW = DTW
pi_DTW = torch.tensor(pi_DTW).long()
C1 = torch.matmul(source_sq, torch.sum(pi_DTW, dim=1)).sum(-1)
print(C1.shape, C1.sum().item())
C2 = torch.matmul(target_sq, torch.sum(pi_DTW.T, dim=1)).sum(-1)
C3 = torch.tensordot(torch.matmul(source, pi_DTW), target, dims=([1, 2], [1, 2]))
res = C1[:, None] + C2[None, :] - 2 * C3
print((torch.tensor(OT).type(torch.float) * (C1[:, None] + C2[None, :])).sum().item(),
      (torch.tensor(OT).type(torch.float) * (2 * C3)).sum().item())
out_CNN = torch.tensor(OT) * res

# Numpy version
out_np_source = source.detach().numpy()
print(out_np_source.sum())
out_np_target = target.detach().numpy()
np_source_sq = out_np_source ** 2
np_target_sq = out_np_target ** 2
print("sq numpy", np_source_sq.sum())
res_np = np.zeros(shape=(200, 200))
np_out_CNN_before = OT * res_np
pi_DTW = DTW
C1 = np.dot(np_source_sq, np.sum(pi_DTW, axis=1)).sum(-1)
print(C1.shape, C1.sum())
C2 = np.dot(np_target_sq, np.sum(pi_DTW.T, axis=1)).sum(-1)
C3 = np.tensordot(np.dot(out_np_source, pi_DTW), out_np_target.transpose(0, -1, 1), axes=([1, 2], [2, 1]))
res_np = C1[:, None] + C2[None, :] - 2 * C3
np_out_CNN = OT * res_np

print(np_out_CNN.sum(), out_CNN.sum().item())
print(np_out_CNN_before.sum(), out_CNN_before.sum().item())
print(res_np.sum(), res.sum().item())
