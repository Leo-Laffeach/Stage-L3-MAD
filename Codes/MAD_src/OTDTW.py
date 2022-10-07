from argparse import _MutuallyExclusiveGroup
from genericpath import exists
from sklearn.manifold import TSNE, MDS
import time
#  from turtle import title

import matplotlib.pyplot as plt
import ot
import tslearn.metrics as tslm
import numpy.random as npr
import numpy as np
import torch
from zmq import device


class OTDTW:
    def __init__(self, X, Y, classe=None, weights_X=None, weights_Y=None, metric="l2", settings=0, classe_unique=None,
                 previous_DTW=None, additional_cost=None, alpha=1.0, beta=0.0):
        if torch.is_tensor(X):
            X = X.cpu().numpy()
            Y = Y.cpu().numpy()
            classe = classe.cpu().numpy()
        self.X = X
        self.Y = Y
        self.alpha = alpha
        self.beta = beta
        self.shapeX = X.shape
        self.shapeY = Y.shape
        self.fst_half = 0
        self.fst_half_iter = 0
        if classe is not None:
            classe = classe.astype(int)
            cl_count = 0
            classe_corrected = np.empty((self.shapeX[0], 1), dtype=int)
            for cl in np.unique(classe):
                classe_corrected[np.where(classe == cl)] = cl_count
                cl_count = cl_count + 1
            self.classe = classe_corrected
            if classe_unique is not None:
                self.classe_unique = classe_unique
            else:
                self.classe_unique = np.unique(self.classe)
        else:
            self.classe = np.zeros((self.shapeX[0], 1), dtype=int)
            self.classe_unique = np.unique(self.classe)
        if self.shapeX[-1] == 1:
            self.one_dim = True
        else:
            self.one_dim = False
        if weights_X is None:
            self.Xa_one = np.ones(self.shapeX[0]) / self.shapeX[0]
        else:
            self.Xa_one = weights_X
        if weights_Y is None:
            self.Ya_one = np.ones(self.shapeY[0]) / self.shapeY[0]
        else:
            self.Ya_one = weights_Y
        
        self.OT_tilde = self.init_OT_matrix(settings)
        self.metric = metric
        self.add_cost = additional_cost
        self.tab_idx = []
        self.dist_OT = []
        self.pi_DTW_idx = []
        self.pi_DTW_path_idx = []
        if self.metric == 'l2':
            self.Xsquared = []
            self.Xsquared_sum = []
            if self.one_dim:
                self.Ysquared = self.Y[:, :, 0] ** 2
            else:
                self.Ysquared = self.Y ** 2
        for cl in self.classe_unique:
            self.tab_idx.append(np.where(self.classe == cl)[0])
            if previous_DTW is None:
                self.pi_DTW_idx.append(self.init_DTW_matrix(settings))
            else:
                self.pi_DTW_idx.append(previous_DTW[cl])
            if self.metric == 'l2':
                if self.one_dim:
                    X2 = self.X[self.tab_idx[cl], :, 0] ** 2
                    X2_sum = np.dot(self.Xa_one[self.tab_idx[cl]], X2)
                else:
                    X2 = self.X[self.tab_idx[cl]] ** 2
                    X2_sum = np.dot(self.Xa_one[self.tab_idx[cl]], X2.transpose(1, 0, -1)).sum(-1)
                self.Xsquared.append(X2)
                self.Xsquared_sum.append(X2_sum[:, None])

    # Initialisation of the OT_tilde matrix

    def init_OT_matrix(self, settings):
        npr.seed(settings)
        # cost_OT = npr.random((self.shapeX[0], self.shapeY[0])) ** 2
        cost_OT = np.ones((self.shapeX[0], self.shapeY[0]))
        OT_tilde = ot.emd(self.Xa_one, self.Ya_one, cost_OT, numItermax=10000000)
        return OT_tilde

    def init_DTW_matrix(self, settings):
        npr.seed(settings)
        DTW_matrix = np.zeros((self.shapeX[1], self.shapeY[1]))
        ts = [0, 0]
        indices_table = [[1, 0], [0, 1], [1, 1]]
        while (ts[0] != self.shapeX[1] - 1) or (ts[1] != self.shapeY[1] - 1):
            DTW_matrix[ts[0], ts[1]] = 1
            if ts[0] == self.shapeX[1] - 1:
                indice_moving = 1
            elif ts[1] == self.shapeY[1] - 1:
                indice_moving = 0
            else:
                if settings == 100:
                    indice_moving = 1
                else:
                    indice_moving = npr.randint(3)
            ts[0] = ts[0] + indices_table[indice_moving][0]
            ts[1] = ts[1] + indices_table[indice_moving][1]
        DTW_matrix[-1, -1] = 1
        return DTW_matrix

    def mat_cost_OT(self):
        mat_cost = np.zeros(shape=(self.shapeX[0], self.shapeY[0]))
        if self.one_dim:
            for cl in self.classe_unique:
                if self.metric == "l2":
                    pi_DTW = self.pi_DTW_idx[cl]
                    C1 = np.dot(self.Xsquared[cl], np.sum(pi_DTW, axis=1))
                    C2 = np.dot(self.Ysquared, np.sum(pi_DTW.T, axis=1))
                    C3 = np.dot(np.dot(self.X[self.tab_idx[cl], :, 0], pi_DTW[:]), self.Y[:, :, 0].T)
                    res = C1[:, None] + C2[None, :] - 2 * C3
                elif self.metric == "l1":
                    m1, m2 = self.get_warp_matrices(cl)
                    C1 = np.dot(self.X[self.tab_idx[cl], :, 0], m1.T)
                    C2 = np.dot(self.Y[:, :, 0], m2.T)
                    C3 = C1[:, None, :] - C2[None, :, :]
                    res = np.absolute(C3).sum(-1)
                mat_cost[self.tab_idx[cl]] = res
        else:
            for cl in self.classe_unique:
                if self.metric == "l2":
                    pi_DTW = self.pi_DTW_idx[cl]
                    C1 = np.dot(self.Xsquared[cl].transpose(0, -1, 1), np.sum(pi_DTW, axis=1)).sum(-1)
                    C2 = np.dot(self.Ysquared.transpose(0, -1, 1), np.sum(pi_DTW.T, axis=1)).sum(-1)
                    C3 = np.tensordot(np.dot(self.X[self.tab_idx[cl]].transpose(0, -1, 1), pi_DTW), self.Y,
                                      axes=([1, 2], [2, 1]))
                    res = C1[:, None] + C2[None, :] - 2 * C3
                elif self.metric == "l1":
                    m1, m2 = self.get_warp_matrices(cl)
                    C1 = np.dot(self.X[self.tab_idx[cl]].transpose(0, -1, 1), m1.T)
                    C2 = np.dot(self.Y.transpose(0, -1, 1), m2.T)
                    C3 = C1[:, None, :] - C2[None, :, :]
                    res = np.absolute(C3).sum(-1).sum(-1)
                mat_cost[self.tab_idx[cl]] = res
        mat_cost /= (self.shapeX[1] + self.shapeY[1]) / 2
        return mat_cost

    def mat_cost_OT_torch(self):

        mat_cost = torch.zeros(size=(self.shapeX[0], self.shapeY[0]))
        if self.one_dim:
            for cl in self.classe_unique:
                if self.metric == "l2":
                    pi_DTW = self.pi_DTW_idx[cl]
                    C1 = torch.matmul(self.Xsquared[cl], torch.sum(pi_DTW, dim=1))
                    C2 = torch.matmul(self.Ysquared, torch.sum(pi_DTW.T, dim=1))
                    C3 = torch.matmul(torch.matmul(self.X[self.tab_idx[cl], :, 0], pi_DTW[:]), self.Y[:, :, 0].T)
                    res = C1[:, None] + C2[None, :] - 2 * C3

                elif self.metric == "l1":
                    m1, m2 = self.get_warp_matrices(cl)
                    C1 = torch.matmul(self.X[self.tab_idx[cl], :, 0], m1.T)
                    C2 = torch.matmul(self.Y[:, :, 0], m2.T)
                    C3 = C1[:, None, :] - C2[None, :, :]
                    res = torch.absolute(C3).sum(-1)
                mat_cost[self.tab_idx[cl]] = res
        else:
            for cl in self.classe_unique:
                if self.metric == "l2":
                    pi_DTW = self.pi_DTW_idx[cl]
                    C1 = torch.matmul(self.Xsquared[cl], torch.sum(pi_DTW, dim=1)).sum(-1)
                    C2 = torch.matmul(self.Ysquared, torch.sum(pi_DTW.T, dim=1)).sum(-1)
                    C3 = torch.tensordot(torch.matmul(self.X[self.tab_idx[cl]], pi_DTW), self.Y, dims=([1, 2], [1, 2]))
                    res = C1[:, None] + C2[None, :] - 2 * C3
                elif self.metric == "l1":
                    m1, m2 = self.get_warp_matrices(cl)
                    C1 = torch.matmul(self.X[self.tab_idx[cl]].transpose(0, -1, 1), m1.T)
                    C2 = torch.matmul(self.Y.transpose(0, -1, 1), m2.T)
                    C3 = C1[:, None, :] - C2[None, :, :]
                    res = torch.absolute(C3).sum(-1).sum(-1)
                mat_cost[self.tab_idx[cl]] = res
        mat_cost /= (self.shapeX[1] + self.shapeY[1]) / 2
        return mat_cost

    def mat_dist_DTW(self, classe_it=None):
        if self.one_dim:
            if classe_it is None:
                OTc = self.OT_tilde
                Xc = self.X[:, :, 0]
            else:
                OTc = self.OT_tilde[self.tab_idx[classe_it]]
                Xc = self.X[self.tab_idx[classe_it], :, 0]
            if self.metric == "l2":
                C2 = np.dot(OTc.sum(axis=0), self.Ysquared)
                C3 = np.dot(np.dot(Xc.T, OTc), self.Y[:, :, 0])
                res = self.Xsquared_sum[classe_it] + C2[None, :] - 2 * C3
            elif self.metric == "l1":
                w1, w2 = self.extended_OT_mat(classe_it)
                C1 = np.dot(w1, Xc)
                C2 = np.dot(w2, self.Y[:, :, 0])
                C3 = np.absolute(C1[:, :, None] - C2[:, None, :])
                res = C3.sum(0)
        else:
            if classe_it is None:
                OTc = self.OT_tilde
                Xc = self.X
            else:
                OTc = self.OT_tilde[self.tab_idx[classe_it]]
                Xc = self.X[self.tab_idx[classe_it]]
            if self.metric == "l2":
                t0 = time.time()
                C2 = np.dot(OTc.sum(axis=0), self.Ysquared.transpose(1, 0, -1)).sum(-1)
                t1 = time.time()
                C31 = np.dot(Xc.T, OTc)
                t1bis = time.time()
                self.fst_half += t1bis - t1
                self.fst_half_iter += 1
                C32 = np.tensordot(C31, self.Y, axes=([0, 2], [2, 0]))
                # C3 = np.tensordot(np.dot(Xc.T, OTc), self.Y, axes=([0, 2], [2, 0]))
                t2 = time.time()
                res = self.Xsquared_sum[classe_it] + C2[None, :] - 2 * C32
                t3 = time.time()
            elif self.metric == "l1":
                w1, w2 = self.extended_OT_mat(classe_it)
                C1 = np.dot(Xc.transpose(-1, 1, 0), w1.T)
                C2 = np.dot(self.Y.transpose(-1, 1, 0), w2.T)
                C3 = np.absolute(C1[:, :, None] - C2[:, None, :])
                res = C3.sum(0).sum(-1)
        res /= (self.shapeX[1] + self.shapeY[1]) / 2
        return res

    def get_warp_matrices(self, cl):

        num_path = int(self.pi_DTW_idx[cl].sum())
        Mx = np.zeros((num_path, self.shapeX[1]))
        My = np.zeros((num_path, self.shapeY[1]))

        k = 0
        l = 0
        for j in range(0, num_path):
            Mx[j, k] = 1
            My[j, l] = 1
            if (k == self.shapeX[1] - 1) & (l != self.shapeY[1] - 1):
                arg1 = -1
                arg2 = self.pi_DTW_idx[cl][k, l + 1]
                arg3 = -1
            if (l == self.shapeY[1] - 1) & (k != self.shapeX[1] - 1):
                arg1 = self.pi_DTW_idx[cl][k + 1, l]
                arg2 = -1
                arg3 = -1
            if (l != self.shapeY[1] - 1) & (k != self.shapeX[1] - 1):
                arg1 = self.pi_DTW_idx[cl][k + 1, l]
                arg2 = self.pi_DTW_idx[cl][k, l + 1]
                arg3 = self.pi_DTW_idx[cl][k + 1, l + 1]

            pos_move = np.argmax((arg1, arg2, arg3))
            if pos_move == 0:
                k = k + 1
            if pos_move == 1:
                l = l + 1
            if pos_move == 2:
                l = l + 1
                k = k + 1
        return Mx, My

    def path2mat(self, path):
        pi_DTW = np.zeros((self.shapeX[1], self.shapeY[1]))
        for i, j in path:
            pi_DTW[i, j] = 1
        return pi_DTW

    def extended_OT_mat(self, classe=None):

        if classe is None:
            lenx = self.shapeX[0]
            OTc = self.OT_tilde
        else:
            lenx = len(self.tab_idx[classe])
            OTc = self.OT_tilde[self.tab_idx[classe]]
        el_t = np.count_nonzero(OTc)

        Wx = np.zeros((el_t, lenx))
        Wy = np.zeros((el_t, self.shapeY[0]))
        index_counting_y = 0
        index_counting_x = 0
        for i in range(0, lenx):
            cnt_z = np.count_nonzero(OTc[i, :])
            nnz = np.nonzero(OTc[i, :])
            Wx[index_counting_x: index_counting_x + cnt_z, i] = OTc[i, nnz]
            index_counting_x = index_counting_x + cnt_z
            for j in nnz[0]:
                Wy[index_counting_y, j] = OTc[i, j]
                index_counting_y = index_counting_y + 1
        return Wx, Wy

    def stopping_criterion(self, last_pi_DTW):
        stop = True
        for cl in self.classe_unique:
            pi_DTW = self.pi_DTW_idx[cl]
            last_DTW = last_pi_DTW[cl]
            if (pi_DTW != last_DTW).any():
                stop = False
        return stop

    def main_training(self, max_init=100, first_step_DTW=True):
        ct_cost_OT = 0
        ct_OT = 0
        ct_cost_DTW = 0
        ct_DTW = 0
        cost = {"Cost": []}
        stop = False
        current_init = 0
        # Begin training
        while stop is not True and current_init < max_init:
            if (current_init != 0) or (first_step_DTW is False):
                t_cost_ot = time.time()
                t0 = time.time()
                Cost_OT_alpha = self.alpha * self.mat_cost_OT()
                t1 = time.time()
                if self.beta != 0:
                    Cost_0T_beta = self.beta * self.add_cost
                    Cost_OT = Cost_OT_alpha + Cost_0T_beta
                else:
                    Cost_OT = Cost_OT_alpha
                t_ot = time.time()
                ct_cost_OT += t_ot - t_cost_ot
                self.OT_tilde = ot.emd(self.Xa_one, self.Ya_one, Cost_OT, numItermax=1000000)
                t_after_ot = time.time()
                ct_OT += t_after_ot - t_ot
                score_OT = np.sum(self.OT_tilde * Cost_OT)
                cost["Cost"].append(score_OT)

            dtw_score = 0
            self.pi_DTW_path_idx = []
            total_cost_DTW = []
            t_before_DTW_cost = time.time()
            for cl in self.classe_unique:
                mat_dist = self.mat_dist_DTW(cl)
                total_cost_DTW.append(mat_dist)
                t_after_DTW_cost = time.time()
                self.fst_half += (t_after_DTW_cost - t_before_DTW_cost)
                ct_cost_DTW += t_after_DTW_cost - t_before_DTW_cost
                Pi_DTW_path, dtw_score_prov = tslm.dtw_path_from_metric(mat_dist, metric="precomputed")
                self.pi_DTW_path_idx.append(Pi_DTW_path)
                Pi_DTW_prov = self.path2mat(Pi_DTW_path)
                self.pi_DTW_idx[cl] = Pi_DTW_prov
                dtw_score += dtw_score_prov
            t_after_DTW = time.time()
            cost["Cost"].append(dtw_score)
            if current_init != 0:
                stop = self.stopping_criterion(last_pi_DTW)
            last_pi_DTW = self.pi_DTW_idx.copy()
            current_init = current_init + 1
        else:
            return self.OT_tilde, self.pi_DTW_idx, Cost_OT, score_OT

    def to_onehot(self, y=None):
        if y is None:
            y = self.classe.squeeze()
        n_values = np.max(y) + 1
        return np.eye(n_values)[y]

    def evaluate(self, train_target_label):
        yt_onehot = self.to_onehot()
        y_pred = np.argmax(np.dot(self.OT_tilde.T, yt_onehot), axis=1)
        accuracy = np.mean(y_pred == train_target_label)
        return accuracy, y_pred


class OTDTW_torch32:
    def __init__(self, X, Y, classe=None, weights_X=None, weights_Y=None, metric="l2", settings=0, classe_unique=None,
                 previous_DTW=None, additional_cost=None, alpha=1.0, beta=1.0, GPU=False):
        self.GPU = GPU
        self.X = X
        self.Y = Y
        self.alpha = alpha
        self.beta = beta
        self.shapeX = X.shape
        self.shapeY = Y.shape
        self.fst_half = 0
        self.fst_half_iter = 0
        if classe is not None:
            classe = classe.type(torch.int32)
            cl_count = 0
            classe_corrected = torch.empty(size=(self.shapeX[0],), dtype=torch.int)
            for cl in torch.unique(classe):
                classe_corrected[classe == cl] = cl_count
                cl_count = cl_count + 1
            self.classe = classe_corrected
            if classe_unique is not None:
                self.classe_unique = classe_unique
            else:
                self.classe_unique = torch.unique(self.classe)
        else:
            self.classe = torch.zeros(size=(self.shapeX[0],), dtype=torch.int)
            self.classe_unique = torch.unique(self.classe)

        if self.shapeX[-1] == 1:
            self.one_dim = True
        else:
            self.one_dim = False
        if weights_X is None:
            self.Xa_one = torch.ones(size=(self.shapeX[0],)) / self.shapeX[0]
        else:
            self.Xa_one = weights_X
        if weights_Y is None:
            self.Ya_one = torch.ones(size=(self.shapeY[0],)) / self.shapeY[0]
        else:
            self.Ya_one = weights_Y

        self.OT_tilde = self.init_OT_matrix(settings)
        self.OT_tilde = torch.tensor(self.OT_tilde).type(torch.float32)
        self.metric = metric
        self.add_cost = additional_cost
        self.tab_idx = []
        self.dist_OT = []
        self.pi_DTW_idx = []
        self.pi_DTW_path_idx = []
        self.device = torch.device("cpu")
        if torch.cuda.is_available():
            self.Xa_one = self.Xa_one.cuda()
            self.Ya_one = self.Ya_one.cuda()
        if self.metric == 'l2':
            self.Xsquared = []
            self.Xsquared_sum = []
            if self.one_dim:
                self.Ysquared = torch.square(self.Y[:, :, 0]).squeeze()
            else:
                self.Ysquared = torch.square(self.Y).squeeze()
        for cl in self.classe_unique:
            if cl == 0:
                self.tab_idx.append((self.classe == 0).nonzero().squeeze())
            else:
                self.tab_idx.append(self.classe.eq(cl).nonzero().squeeze())
            if torch.cuda.is_available():
                self.tab_idx[cl] = self.tab_idx[cl].cuda()
            if previous_DTW is None:
                self.pi_DTW_idx.append(self.init_DTW_matrix(settings))
            else:
                if torch.cuda.is_available():
                    previous_DTW[cl] = previous_DTW[cl].cuda()
                self.pi_DTW_idx.append(previous_DTW[cl])
            if self.metric == 'l2':
                if self.one_dim:
                    X2 = torch.square(self.X[self.tab_idx[cl], :, 0])
                    X2_sum = torch.matmul(self.Xa_one[self.tab_idx[cl]], X2)
                else:
                    X2 = torch.square(self.X[self.tab_idx[cl]])
                    X2_sum = torch.matmul(self.Xa_one[self.tab_idx[cl]], X2.transpose(0, 1)).sum(-1)
                self.Xsquared.append(X2)
                self.Xsquared_sum.append(X2_sum[:, None])

    # Initialisation of the OT_tilde matrix

    def init_OT_matrix(self, settings):
        torch.manual_seed(settings)
        # cost_OT = torch.square(torch.rand(size=(self.shapeX[0], self.shapeY[0]))).squeeze()
        cost_OT = torch.ones(size=(self.shapeX[0], self.shapeY[0]))
        OT_tilde = ot.emd(self.Xa_one, self.Ya_one, cost_OT, numItermax=10000000)
        return OT_tilde

    def init_DTW_matrix(self, settings):
        torch.manual_seed(settings)
        DTW_matrix = torch.zeros(size=(self.shapeX[1], self.shapeY[1]))
        ts = [0, 0]
        indices_table = [[1, 0], [0, 1], [1, 1]]
        while (ts[0] != self.shapeX[1] - 1) or (ts[1] != self.shapeY[1] - 1):
            DTW_matrix[ts[0], ts[1]] = 1
            if ts[0] == self.shapeX[1] - 1:
                indice_moving = 1
            elif ts[1] == self.shapeY[1] - 1:
                indice_moving = 0
            else:
                if settings == 100:
                    indice_moving = 1
                else:
                    indice_moving = torch.randint(0, 3, (1,))
            ts[0] = ts[0] + indices_table[indice_moving][0]
            ts[1] = ts[1] + indices_table[indice_moving][1]
        DTW_matrix[-1, -1] = 1
        if torch.cuda.is_available():
            DTW_matrix = DTW_matrix.cuda()
        return DTW_matrix

    def mat_cost_OT(self):
        if torch.cuda.is_available():
            mat_cost = torch.zeros(size=(self.shapeX[0], self.shapeY[0]), device="cuda:0")
        else:
            mat_cost = torch.zeros(size=(self.shapeX[0], self.shapeY[0]))
        if self.one_dim:
            for cl in self.classe_unique:
                if self.metric == "l2":
                    pi_DTW = self.pi_DTW_idx[cl]
                    C1 = torch.matmul(self.Xsquared[cl], torch.sum(pi_DTW, dim=1))
                    C2 = torch.matmul(self.Ysquared, torch.sum(pi_DTW.T, dim=1))
                    C3 = torch.matmul(torch.matmul(self.X[self.tab_idx[cl], :, 0], pi_DTW[:]), self.Y[:, :, 0].T)
                    res = C1[:, None] + C2[None, :] - 2 * C3

                elif self.metric == "l1":
                    m1, m2 = self.get_warp_matrices(cl)
                    C1 = torch.matmul(self.X[self.tab_idx[cl], :, 0], m1.T)
                    C2 = torch.matmul(self.Y[:, :, 0], m2.T)
                    C3 = C1[:, None, :] - C2[None, :, :]
                    res = torch.absolute(C3).sum(-1)
                mat_cost[self.tab_idx[cl]] = res
        else:
            for cl in self.classe_unique:
                if self.metric == "l2":
                    pi_DTW = self.pi_DTW_idx[cl]
                    C1 = torch.matmul(self.Xsquared[cl].transpose(1, -1), torch.sum(pi_DTW, dim=1)).sum(-1)
                    C2 = torch.matmul(self.Ysquared.transpose(1, -1), torch.sum(pi_DTW.T, dim=1)).sum(-1)
                    C3 = torch.tensordot(torch.matmul(self.X[self.tab_idx[cl]].transpose(1, -1), pi_DTW), self.Y,
                                         dims=([1, 2], [2, 1]))
                    res = C1[:, None] + C2[None, :] - 2 * C3
                elif self.metric == "l1":
                    m1, m2 = self.get_warp_matrices(cl)
                    C1 = torch.matmul(self.X[self.tab_idx[cl]].transpose(1, -1), m1.T)
                    C2 = torch.matmul(self.Y.transpose(1, -1), m2.T)
                    C3 = C1[:, None, :] - C2[None, :, :]
                    res = torch.absolute(C3).sum(-1).sum(-1)
                mat_cost[self.tab_idx[cl]] = res
        mat_cost /= (self.shapeX[1] + self.shapeY[1]) / 2
        return mat_cost

    def mat_dist_DTW(self, classe_it=None):
        if self.one_dim:
            if classe_it is None:
                OTc = self.OT_tilde
                Xc = self.X[:, :, 0]
            else:
                OTc = self.OT_tilde[self.tab_idx[classe_it]]
                Xc = self.X[self.tab_idx[classe_it], :, 0]
            if self.metric == "l2":
                C2 = torch.matmul(OTc.sum(axis=0), self.Ysquared)
                C3 = torch.matmul(torch.matmul(Xc.T, OTc), self.Y[:, :, 0])
                res = self.Xsquared_sum[classe_it] + C2[None, :] - 2 * C3
            elif self.metric == "l1":
                w1, w2 = self.extended_OT_mat(classe_it)
                C1 = torch.matmul(w1, Xc)
                C2 = torch.matmul(w2, self.Y[:, :, 0])
                C3 = torch.absolute(C1[:, :, None] - C2[:, None, :])
                res = C3.sum(0)
        else:
            if classe_it is None:
                OTc = self.OT_tilde
                Xc = self.X
            else:
                OTc = self.OT_tilde[self.tab_idx[classe_it]]
                Xc = self.X[self.tab_idx[classe_it]]
            if self.metric == "l2":
                C2 = torch.matmul(OTc.sum(0), self.Ysquared.transpose(0, 1)).sum(-1)
                C31 = torch.matmul(Xc.T, OTc)
                C32 = torch.tensordot(C31, self.Y, dims=([0, 2], [2, 0]))
                res = self.Xsquared_sum[classe_it] + C2[None, :] - 2 * C32
            elif self.metric == "l1":
                w1, w2 = self.extended_OT_mat(classe_it)
                C1 = torch.matmul(Xc.transpose(0, -1), w1.T)
                C2 = torch.matmul(self.Y.transpose(0, -1), w2.T)
                C3 = torch.absolute(C1[:, :, None] - C2[:, None, :])
                res = C3.sum(0).sum(-1)
        res /= (self.shapeX[1] + self.shapeY[1]) / 2
        return res

    def get_warp_matrices(self, cl):

        num_path = int(self.pi_DTW_idx[cl].sum())
        Mx = torch.zeros((num_path, self.shapeX[1]))
        My = torch.zeros((num_path, self.shapeY[1]))

        k = 0
        l = 0
        for j in range(0, num_path):
            Mx[j, k] = 1
            My[j, l] = 1
            if (k == self.shapeX[1] - 1) & (l != self.shapeY[1] - 1):
                arg1 = -1
                arg2 = self.pi_DTW_idx[cl][k, l + 1]
                arg3 = -1
            if (l == self.shapeY[1] - 1) & (k != self.shapeX[1] - 1):
                arg1 = self.pi_DTW_idx[cl][k + 1, l]
                arg2 = -1
                arg3 = -1
            if (l != self.shapeY[1] - 1) & (k != self.shapeX[1] - 1):
                arg1 = self.pi_DTW_idx[cl][k + 1, l]
                arg2 = self.pi_DTW_idx[cl][k, l + 1]
                arg3 = self.pi_DTW_idx[cl][k + 1, l + 1]

            pos_move = np.argmax((arg1, arg2, arg3))
            if pos_move == 0:
                k = k + 1
            if pos_move == 1:
                l = l + 1
            if pos_move == 2:
                l = l + 1
                k = k + 1
        return Mx, My

    def path2mat(self, path):
        if torch.cuda.is_available():
            pi_DTW = torch.zeros((self.shapeX[1], self.shapeY[1]), device="cuda:0")
        else:
            pi_DTW = torch.zeros((self.shapeX[1], self.shapeY[1]))
        for i, j in path:
            pi_DTW[i, j] = 1
        return pi_DTW

    def extended_OT_mat(self, classe=None):

        if classe is None:
            lenx = self.shapeX[0]
            OTc = self.OT_tilde
        else:
            lenx = len(self.tab_idx[classe])
            OTc = self.OT_tilde[self.tab_idx[classe]]
        el_t = torch.count_nonzero(OTc)

        Wx = torch.zeros(size=(el_t, lenx))
        Wy = torch.zeros(size=(el_t, self.shapeY[0]))
        index_counting_y = 0
        index_counting_x = 0
        for i in range(0, lenx):
            cnt_z = torch.count_nonzero(OTc[i, :])
            nnz = torch.nonzero(OTc[i, :])
            Wx[index_counting_x: index_counting_x + cnt_z, i] = OTc[i, nnz]
            index_counting_x = index_counting_x + cnt_z
            for j in nnz[0]:
                Wy[index_counting_y, j] = OTc[i, j]
                index_counting_y = index_counting_y + 1
        return Wx, Wy

    def stopping_criterion(self, last_pi_DTW):
        stop = True
        for cl in self.classe_unique:
            pi_DTW = self.pi_DTW_idx[cl]
            last_DTW = last_pi_DTW[cl]
            if (pi_DTW != last_DTW).any():
                stop = False
        return stop

    def main_training(self, max_init=100, first_step_DTW=True):
        cost = {"Cost": []}
        stop = False
        current_init = 0
        # Begin training
        while stop is not True and current_init < max_init:
            if (current_init != 0) or (first_step_DTW is False):
                Cost_OT_alpha = self.alpha * self.mat_cost_OT()
                if self.beta != 0:
                    Cost_0T_beta = self.beta * self.add_cost
                    Cost_OT = Cost_OT_alpha + Cost_0T_beta
                else:
                    Cost_OT = Cost_OT_alpha

                OT_tilde = ot.emd(self.Xa_one, self.Ya_one, Cost_OT, numItermax=1000000)
                OT_tilde = torch.tensor(OT_tilde).type(torch.float32)
                self.OT_tilde = OT_tilde.type(torch.float32)
                score_OT = torch.sum(self.OT_tilde * Cost_OT)
                cost["Cost"].append(score_OT)

            dtw_score = 0
            self.pi_DTW_path_idx = []
            total_cost_dtw = []
            for cl in self.classe_unique:
                mat_dist = self.mat_dist_DTW(cl)
                total_cost_dtw.append(mat_dist)
                Pi_DTW_path, dtw_score_prov = torch_dtw(mat_dist)
                self.pi_DTW_path_idx.append(Pi_DTW_path)
                Pi_DTW_prov = self.path2mat(Pi_DTW_path)
                self.pi_DTW_idx[cl] = Pi_DTW_prov
                dtw_score += dtw_score_prov
            cost["Cost"].append(dtw_score)
            if current_init != 0:
                stop = self.stopping_criterion(last_pi_DTW)
            last_pi_DTW = self.pi_DTW_idx.copy()
            current_init = current_init + 1
        else:
            return self.OT_tilde, self.pi_DTW_idx, Cost_OT, score_OT

    def to_onehot(self, y=None):
        if y is None:
            y = self.classe.numpy().astype(int).squeeze()
        n_values = np.max(y) + 1
        return np.eye(n_values)[y]

    def evaluate(self, train_target_label):
        yt_onehot = self.to_onehot()
        y_pred = np.argmax(np.dot(self.OT.T, yt_onehot), axis=1)
        accuracy = np.mean(y_pred == train_target_label.numpy().astype(int))
        return accuracy, y_pred


class OTDTW_CPU:
    def __init__(self, X, Y, classe=None, weights_X=None, weights_Y=None, metric="l2", settings=0, classe_unique=None,
                 previous_DTW=None, additional_cost=None, alpha=1.0, beta=1.0, GPU=False):
        self.GPU = GPU
        if self.GPU:
            self.X = X
            self.Y = Y
        elif self.GPU is False:
            self.X = X.cpu().detach()
            self.Y = Y.cpu().detach()
        self.alpha = alpha
        self.beta = beta
        self.shapeX = X.shape
        self.shapeY = Y.shape
        self.fst_half = 0
        self.fst_half_iter = 0
        if classe is not None:
            classe = classe.type(torch.int32)
            cl_count = 0
            classe_corrected = torch.empty(size=(self.shapeX[0],), dtype=torch.int)
            for cl in torch.unique(classe):
                classe_corrected[classe == cl] = cl_count
                cl_count = cl_count + 1
            self.classe = classe_corrected
            if classe_unique is not None:
                self.classe_unique = classe_unique
            else:
                self.classe_unique = torch.unique(self.classe)
        else:
            self.classe = torch.zeros(size=(self.shapeX[0],), dtype=torch.int)
            self.classe_unique = torch.unique(self.classe)

        if self.shapeX[-1] == 1:
            self.one_dim = True
        else:
            self.one_dim = False
        if weights_X is None:
            self.Xa_one = torch.ones(size=(self.shapeX[0],)) / self.shapeX[0]
        else:
            self.Xa_one = weights_X
        if weights_Y is None:
            self.Ya_one = torch.ones(size=(self.shapeY[0],)) / self.shapeY[0]
        else:
            self.Ya_one = weights_Y

        self.OT_tilde = self.init_OT_matrix(settings)
        self.metric = metric
        self.add_cost = additional_cost
        if (self.add_cost is not None) & (self.GPU is False):
            self.add_cost = self.add_cost.cpu().detach()
        self.tab_idx = []
        self.dist_OT = []
        self.pi_DTW_idx = []
        self.pi_DTW_path_idx = []
        if (torch.cuda.is_available()) and self.GPU:
            self.Xa_one = self.Xa_one.cuda()
            self.Ya_one = self.Ya_one.cuda()
        if self.metric == 'l2':
            self.Xsquared = []
            self.Xsquared_sum = []
            if self.one_dim:
                self.Ysquared = torch.square(self.Y[:, :, 0]).squeeze()
            else:
                self.Ysquared = torch.square(self.Y).squeeze()
        for cl in self.classe_unique:
            if cl == 0:
                self.tab_idx.append((self.classe == 0).nonzero().squeeze())
            else:
                self.tab_idx.append(self.classe.eq(cl).nonzero().squeeze())
            if (torch.cuda.is_available()) and self.GPU:
                self.tab_idx[cl] = self.tab_idx[cl].cuda()
            if previous_DTW is None:
                self.pi_DTW_idx.append(self.init_DTW_matrix(settings))
            else:
                if (torch.cuda.is_available()) and self.GPU:
                    previous_DTW[cl] = previous_DTW[cl].cuda()
                self.pi_DTW_idx.append(previous_DTW[cl])
            if self.metric == 'l2':
                if self.one_dim:
                    X2 = torch.square(self.X[self.tab_idx[cl], :, 0])
                    X2_sum = torch.matmul(self.Xa_one[self.tab_idx[cl]], X2)
                else:
                    X2 = torch.square(self.X[self.tab_idx[cl]])
                    X2_sum = torch.matmul(self.Xa_one[self.tab_idx[cl]], X2.transpose(0, 1)).sum(-1)
                self.Xsquared.append(X2)
                self.Xsquared_sum.append(X2_sum[:, None])

    # Initialisation of the OT_tilde matrix

    def init_OT_matrix(self, settings):
        torch.manual_seed(settings)
        # cost_OT = torch.square(torch.rand(size=(self.shapeX[0], self.shapeY[0]))).squeeze()
        cost_OT = torch.ones(size=(self.shapeX[0], self.shapeY[0]))
        OT_tilde = ot.emd(self.Xa_one, self.Ya_one, cost_OT, numItermax=10000000)
        # OT_tilde = torch.from_numpy(OT_tilde)
        return OT_tilde

    def init_DTW_matrix(self, settings):
        torch.manual_seed(settings)
        DTW_matrix = torch.zeros(size=(self.shapeX[1], self.shapeY[1]))
        ts = [0, 0]
        indices_table = [[1, 0], [0, 1], [1, 1]]
        while (ts[0] != self.shapeX[1] - 1) or (ts[1] != self.shapeY[1] - 1):
            DTW_matrix[ts[0], ts[1]] = 1
            if ts[0] == self.shapeX[1] - 1:
                indice_moving = 1
            elif ts[1] == self.shapeY[1] - 1:
                indice_moving = 0
            else:
                if settings == 100:
                    indice_moving = 1
                else:
                    indice_moving = torch.randint(0, 3, (1,))
            ts[0] = ts[0] + indices_table[indice_moving][0]
            ts[1] = ts[1] + indices_table[indice_moving][1]
        DTW_matrix[-1, -1] = 1
        if (torch.cuda.is_available()) and self.GPU:
            DTW_matrix = DTW_matrix.cuda()
        return DTW_matrix

    def mat_cost_OT(self):

        mat_cost = torch.zeros(size=(self.shapeX[0], self.shapeY[0]))
        if (torch.cuda.is_available()) and self.GPU:
            mat_cost = mat_cost.cuda()
        if self.one_dim:
            for cl in self.classe_unique:
                if self.metric == "l2":
                    pi_DTW = self.pi_DTW_idx[cl]
                    C1 = torch.matmul(self.Xsquared[cl], torch.sum(pi_DTW, dim=1))
                    C2 = torch.matmul(self.Ysquared, torch.sum(pi_DTW.T, dim=1))
                    C3 = torch.matmul(torch.matmul(self.X[self.tab_idx[cl], :, 0], pi_DTW[:]), self.Y[:, :, 0].T)
                    res = C1[:, None] + C2[None, :] - 2 * C3

                elif self.metric == "l1":
                    m1, m2 = self.get_warp_matrices(cl)
                    C1 = torch.matmul(self.X[self.tab_idx[cl], :, 0], m1.T)
                    C2 = torch.matmul(self.Y[:, :, 0], m2.T)
                    C3 = C1[:, None, :] - C2[None, :, :]
                    res = torch.absolute(C3).sum(-1)
                mat_cost[self.tab_idx[cl]] = res
        else:
            for cl in self.classe_unique:
                if self.metric == "l2":
                    pi_DTW = self.pi_DTW_idx[cl]
                    C1 = torch.matmul(self.Xsquared[cl].transpose(1, -1), torch.sum(pi_DTW, dim=1)).sum(-1)
                    C2 = torch.matmul(self.Ysquared.transpose(1, -1), torch.sum(pi_DTW.T, dim=1)).sum(-1)                    
                    C3 = torch.tensordot(torch.matmul(self.X[self.tab_idx[cl]].transpose(1, -1), pi_DTW), self.Y,
                                         dims=([1, 2], [2, 1]))
                    res = C1[:, None] + C2[None, :] - 2 * C3
                elif self.metric == "l1":
                    m1, m2 = self.get_warp_matrices(cl)
                    C1 = torch.matmul(self.X[self.tab_idx[cl]].transpose(1, -1), m1.T)
                    C2 = torch.matmul(self.Y.transpose(1, -1), m2.T)
                    C3 = C1[:, None, :] - C2[None, :, :]
                    res = torch.absolute(C3).sum(-1).sum(-1)
                mat_cost[self.tab_idx[cl]] = res
        mat_cost /= (self.shapeX[1] + self.shapeY[1]) / 2
        return mat_cost

    def mat_dist_DTW(self, classe_it=None):
        if self.one_dim:
            if classe_it is None:
                OTc = self.OT_tilde
                Xc = self.X[:, :, 0]
            else:
                OTc = self.OT_tilde[self.tab_idx[classe_it]]
                Xc = self.X[self.tab_idx[classe_it], :, 0]
            if self.metric == "l2":
                C2 = torch.matmul(OTc.sum(axis=0), self.Ysquared)
                C3 = torch.matmul(torch.matmul(Xc.T, OTc), self.Y[:, :, 0])
                res = self.Xsquared_sum[classe_it] + C2[None, :] - 2 * C3
            elif self.metric == "l1":
                w1, w2 = self.extended_OT_mat(classe_it)
                C1 = torch.matmul(w1, Xc)
                C2 = torch.matmul(w2, self.Y[:, :, 0])
                C3 = torch.absolute(C1[:, :, None] - C2[:, None, :])
                res = C3.sum(0)
        else:
            if classe_it is None:
                OTc = self.OT_tilde
                Xc = self.X
            else:
                OTc = self.OT_tilde[self.tab_idx[classe_it]]
                Xc = self.X[self.tab_idx[classe_it]]
            if self.metric == "l2":
                C2 = torch.matmul(OTc.sum(0), self.Ysquared.transpose(0, 1)).sum(-1)
                C31 = torch.matmul(Xc.T, OTc)
                C32 = torch.tensordot(C31, self.Y, dims=([0, 2], [2, 0]))
                res = self.Xsquared_sum[classe_it] + C2[None, :] - 2 * C32
            elif self.metric == "l1":
                w1, w2 = self.extended_OT_mat(classe_it)
                C1 = torch.matmul(Xc.transpose(0, -1), w1.T)
                C2 = torch.matmul(self.Y.transpose(0, -1), w2.T)
                C3 = torch.absolute(C1[:, :, None] - C2[:, None, :])
                res = C3.sum(0).sum(-1)
        res /= (self.shapeX[1] + self.shapeY[1]) / 2
        return res

    def get_warp_matrices(self, cl):

        num_path = int(self.pi_DTW_idx[cl].sum())
        Mx = torch.zeros((num_path, self.shapeX[1]))
        My = torch.zeros((num_path, self.shapeY[1]))

        k = 0
        l = 0
        for j in range(0, num_path):
            Mx[j, k] = 1
            My[j, l] = 1
            if (k == self.shapeX[1] - 1) & (l != self.shapeY[1] - 1):
                arg1 = -1
                arg2 = self.pi_DTW_idx[cl][k, l + 1]
                arg3 = -1
            if (l == self.shapeY[1] - 1) & (k != self.shapeX[1] - 1):
                arg1 = self.pi_DTW_idx[cl][k + 1, l]
                arg2 = -1
                arg3 = -1
            if (l != self.shapeY[1] - 1) & (k != self.shapeX[1] - 1):
                arg1 = self.pi_DTW_idx[cl][k + 1, l]
                arg2 = self.pi_DTW_idx[cl][k, l + 1]
                arg3 = self.pi_DTW_idx[cl][k + 1, l + 1]

            pos_move = np.argmax((arg1, arg2, arg3))
            if pos_move == 0:
                k = k + 1
            if pos_move == 1:
                l = l + 1
            if pos_move == 2:
                l = l + 1
                k = k + 1
        return Mx, My

    def path2mat(self, path):
        pi_DTW = torch.zeros((self.shapeX[1], self.shapeY[1]))
        for i, j in path:
            pi_DTW[i, j] = 1
        return pi_DTW

    def extended_OT_mat(self, classe=None):

        if classe is None:
            lenx = self.shapeX[0]
            OTc = self.OT_tilde
        else:
            lenx = len(self.tab_idx[classe])
            OTc = self.OT_tilde[self.tab_idx[classe]]
        el_t = torch.count_nonzero(OTc)

        Wx = torch.zeros(size=(el_t, lenx))
        Wy = torch.zeros(size=(el_t, self.shapeY[0]))
        index_counting_y = 0
        index_counting_x = 0
        for i in range(0, lenx):
            cnt_z = torch.count_nonzero(OTc[i, :])
            nnz = torch.nonzero(OTc[i, :])
            Wx[index_counting_x: index_counting_x + cnt_z, i] = OTc[i, nnz]
            index_counting_x = index_counting_x + cnt_z
            for j in nnz[0]:
                Wy[index_counting_y, j] = OTc[i, j]
                index_counting_y = index_counting_y + 1
        return Wx, Wy

    def stopping_criterion(self, last_pi_DTW):
        stop = True
        for cl in self.classe_unique:
            pi_DTW = self.pi_DTW_idx[cl]
            last_DTW = last_pi_DTW[cl]
            if (pi_DTW != last_DTW).any():
                stop = False
        return stop

    def main_training(self, max_init=100, first_step_DTW=True):
        ct_cost_OT = 0
        ct_OT = 0
        ct_cost_DTW = 0
        ct_DTW = 0
        cost = {"Cost": []}
        stop = False
        current_init = 0
        # Begin training
        prop_show = []
        for cl in range(0, len(torch.unique(self.classe))):
            prop_show.append(round(torch.sum(self.Xa_one[self.classe == cl]).item(), 4))
        while stop is not True and current_init < max_init:
            if (current_init != 0) or (first_step_DTW is False):
                t_cost_ot = time.time()
                Cost_OT_alpha = self.alpha * self.mat_cost_OT()
                t11 = time.time()
                if self.beta != 0:
                    Cost_0T_beta = self.beta * self.add_cost
                    Cost_OT = Cost_OT_alpha + Cost_0T_beta
                else:
                    Cost_OT = Cost_OT_alpha

                t_ot = time.time()
                ct_cost_OT += t_ot - t_cost_ot
                OT_tilde = ot.emd(self.Xa_one.cpu(), self.Ya_one.cpu(), Cost_OT.cpu(), numItermax=1000000)
                if (torch.cuda.is_available()) and self.GPU:
                    OT_tilde = OT_tilde.cuda()
                self.OT_tilde = OT_tilde
                t_after_ot = time.time()
                ct_OT += t_after_ot - t_ot
                score_OT = torch.sum(self.OT_tilde * Cost_OT)
                cost["Cost"].append(score_OT)

            dtw_score = 0
            self.pi_DTW_path_idx = []
            total_cost_dtw = []
            for cl in self.classe_unique:
                t_before_DTW_cost = time.time()
                mat_dist = self.mat_dist_DTW(cl)
                total_cost_dtw.append(mat_dist)
                t_after_DTW_cost = time.time()
                self.fst_half += (t_after_DTW_cost - t_before_DTW_cost)
                ct_cost_DTW += t_after_DTW_cost - t_before_DTW_cost
                Pi_DTW_path, dtw_score_prov = torch_dtw_CPU(mat_dist)
                t_after_DTW = time.time()
                ct_DTW += t_after_DTW - t_after_DTW_cost
                self.pi_DTW_path_idx.append(Pi_DTW_path)
                Pi_DTW_prov = self.path2mat(Pi_DTW_path)
                if (torch.cuda.is_available()) and self.GPU:
                    Pi_DTW_prov = Pi_DTW_prov.cuda()
                self.pi_DTW_idx[cl] = Pi_DTW_prov
                dtw_score += dtw_score_prov
            cost["Cost"].append(dtw_score)
            if current_init != 0:
                stop = self.stopping_criterion(last_pi_DTW)
            last_pi_DTW = self.pi_DTW_idx.copy()
            current_init = current_init + 1
        else:
            return self.OT_tilde, self.pi_DTW_idx, Cost_OT, score_OT

    def to_onehot(self, y=None):
        if y is None:
            y = self.classe.numpy().astype(int)
        n_values = np.max(y) + 1
        return np.eye(n_values)[y]

    def evaluate(self, train_target_label):
        yt_onehot = self.to_onehot()
        y_pred = np.argmax(np.dot(self.OT_tilde.T, yt_onehot), axis=1)
        accuracy = np.mean(y_pred == train_target_label.numpy().astype(int))
        return accuracy, y_pred


class OTDTW_CPU_UNBALANCED:
    def __init__(self, X, Y, classe=None, weights_X=None, weights_Y=None, metric="l2", settings=0, classe_unique=None,
                 previous_DTW=None, additional_cost=None, alpha=1.0, beta=1.0, GPU=False):
        self.GPU = GPU
        if self.GPU:
            self.X = X
            self.Y = Y
        elif self.GPU is False:
            self.X = X.cpu().detach()
            self.Y = Y.cpu().detach()
        self.alpha = alpha
        self.beta = beta
        self.shapeX = X.shape
        self.shapeY = Y.shape
        self.fst_half = 0
        self.fst_half_iter = 0
        if classe is not None:
            classe = classe.type(torch.int32)
            cl_count = 0
            """classe_corrected = torch.empty(size=(self.shapeX[0],), dtype=torch.int)
            for cl in torch.unique(classe):
                classe_corrected[classe == cl] = cl_count
                cl_count = cl_count + 1
            self.classe = classe_corrected"""
            self.classe = classe
            if classe_unique is not None:
                self.classe_unique = classe_unique
            else:
                self.classe_unique = torch.unique(self.classe)
        else:
            self.classe = torch.zeros(size=(self.shapeX[0],), dtype=torch.int)
            self.classe_unique = torch.unique(self.classe)

        if self.shapeX[-1] == 1:
            self.one_dim = True
        else:
            self.one_dim = False
        if weights_X is None:
            self.Xa_one = torch.ones(size=(self.shapeX[0],)) / self.shapeX[0]
        else:
            self.Xa_one = weights_X
        if weights_Y is None:
            self.Ya_one = torch.ones(size=(self.shapeY[0],)) / self.shapeY[0]
        else:
            self.Ya_one = weights_Y

        self.OT_tilde = self.init_OT_matrix(settings)
        self.metric = metric
        self.add_cost = additional_cost
        if (self.add_cost is not None) & (self.GPU is False):
            self.add_cost = self.add_cost.cpu().detach()
        self.tab_idx = []
        self.dist_OT = []
        self.pi_DTW_idx = []
        self.pi_DTW_path_idx = []
        if (torch.cuda.is_available()) and self.GPU:
            self.Xa_one = self.Xa_one.cuda()
            self.Ya_one = self.Ya_one.cuda()
        if self.metric == 'l2':
            self.Xsquared = []
            self.Xsquared_sum = []
            if self.one_dim:
                self.Ysquared = torch.square(self.Y[:, :, 0]).squeeze()
            else:
                self.Ysquared = torch.square(self.Y).squeeze()
        for cl in self.classe_unique:
            if cl == 0:
                self.tab_idx.append((self.classe == 0).nonzero().squeeze())
            else:
                self.tab_idx.append(self.classe.eq(cl).nonzero().squeeze())
            if (torch.cuda.is_available()) and self.GPU:
                self.tab_idx[cl] = self.tab_idx[cl].cuda()
            if previous_DTW is None:
                self.pi_DTW_idx.append(self.init_DTW_matrix(settings))
            else:
                if (torch.cuda.is_available()) and self.GPU:
                    previous_DTW[cl] = previous_DTW[cl].cuda()
                self.pi_DTW_idx.append(previous_DTW[cl])
            if self.metric == 'l2':
                if self.one_dim:
                    X2 = torch.square(self.X[self.tab_idx[cl], :, 0])
                    X2_sum = torch.matmul(self.Xa_one[self.tab_idx[cl]], X2)
                else:
                    X2 = torch.square(self.X[self.tab_idx[cl]])
                    X2_sum = torch.matmul(self.Xa_one[self.tab_idx[cl]], X2.transpose(0, 1)).sum(-1)
                self.Xsquared.append(X2)
                self.Xsquared_sum.append(X2_sum[:, None])

    # Initialisation of the OT_tilde matrix

    def init_OT_matrix(self, settings):
        torch.manual_seed(settings)
        # cost_OT = torch.square(torch.rand(size=(self.shapeX[0], self.shapeY[0]))).squeeze()
        cost_OT = torch.ones(size=(self.shapeX[0], self.shapeY[0]))
        OT_tilde = ot.emd(self.Xa_one, self.Ya_one, cost_OT, numItermax=10000000)
        # OT_tilde = torch.from_numpy(OT_tilde)
        return OT_tilde

    def init_DTW_matrix(self, settings):
        torch.manual_seed(settings)
        DTW_matrix = torch.zeros(size=(self.shapeX[1], self.shapeY[1]))
        ts = [0, 0]
        indices_table = [[1, 0], [0, 1], [1, 1]]
        while (ts[0] != self.shapeX[1] - 1) or (ts[1] != self.shapeY[1] - 1):
            DTW_matrix[ts[0], ts[1]] = 1
            if ts[0] == self.shapeX[1] - 1:
                indice_moving = 1
            elif ts[1] == self.shapeY[1] - 1:
                indice_moving = 0
            else:
                if settings == 100:
                    indice_moving = 1
                else:
                    indice_moving = torch.randint(0, 3, (1,))
            ts[0] = ts[0] + indices_table[indice_moving][0]
            ts[1] = ts[1] + indices_table[indice_moving][1]
        DTW_matrix[-1, -1] = 1
        if (torch.cuda.is_available()) and self.GPU:
            DTW_matrix = DTW_matrix.cuda()
        return DTW_matrix

    def mat_cost_OT(self):

        mat_cost = torch.zeros(size=(self.shapeX[0], self.shapeY[0]))
        if (torch.cuda.is_available()) and self.GPU:
            mat_cost = mat_cost.cuda()
        if self.one_dim:
            for cl in self.classe_unique:
                if self.metric == "l2":
                    pi_DTW = self.pi_DTW_idx[cl]
                    C1 = torch.matmul(self.Xsquared[cl], torch.sum(pi_DTW, dim=1))
                    C2 = torch.matmul(self.Ysquared, torch.sum(pi_DTW.T, dim=1))
                    C3 = torch.matmul(torch.matmul(self.X[self.tab_idx[cl], :, 0], pi_DTW[:]), self.Y[:, :, 0].T)
                    res = C1[:, None] + C2[None, :] - 2 * C3

                elif self.metric == "l1":
                    m1, m2 = self.get_warp_matrices(cl)
                    C1 = torch.matmul(self.X[self.tab_idx[cl], :, 0], m1.T)
                    C2 = torch.matmul(self.Y[:, :, 0], m2.T)
                    C3 = C1[:, None, :] - C2[None, :, :]
                    res = torch.absolute(C3).sum(-1)
                mat_cost[self.tab_idx[cl]] = res
        else:
            for cl in self.classe_unique:
                if self.metric == "l2":
                    pi_DTW = self.pi_DTW_idx[cl]
                    C1 = torch.matmul(self.Xsquared[cl].transpose(1, -1), torch.sum(pi_DTW, dim=1)).sum(-1)
                    C2 = torch.matmul(self.Ysquared.transpose(1, -1), torch.sum(pi_DTW.T, dim=1)).sum(-1)
                    C3 = torch.tensordot(torch.matmul(self.X[self.tab_idx[cl]].transpose(1, -1), pi_DTW), self.Y,
                                         dims=([1, 2], [2, 1]))
                    res = C1[:, None] + C2[None, :] - 2 * C3
                elif self.metric == "l1":
                    m1, m2 = self.get_warp_matrices(cl)
                    C1 = torch.matmul(self.X[self.tab_idx[cl]].transpose(1, -1), m1.T)
                    C2 = torch.matmul(self.Y.transpose(1, -1), m2.T)
                    C3 = C1[:, None, :] - C2[None, :, :]
                    res = torch.absolute(C3).sum(-1).sum(-1)
                mat_cost[self.tab_idx[cl]] = res
        mat_cost /= (self.shapeX[1] + self.shapeY[1]) / 2
        return mat_cost

    def mat_dist_DTW(self, classe_it=None):
        if self.one_dim:
            if classe_it is None:
                OTc = self.OT_tilde
                Xc = self.X[:, :, 0]
            else:
                OTc = self.OT_tilde[self.tab_idx[classe_it]]
                Xc = self.X[self.tab_idx[classe_it], :, 0]
            if self.metric == "l2":
                C2 = torch.matmul(OTc.sum(axis=0), self.Ysquared)
                C3 = torch.matmul(torch.matmul(Xc.T, OTc), self.Y[:, :, 0])
                res = self.Xsquared_sum[classe_it] + C2[None, :] - 2 * C3
            elif self.metric == "l1":
                w1, w2 = self.extended_OT_mat(classe_it)
                C1 = torch.matmul(w1, Xc)
                C2 = torch.matmul(w2, self.Y[:, :, 0])
                C3 = torch.absolute(C1[:, :, None] - C2[:, None, :])
                res = C3.sum(0)
        else:
            if classe_it is None:
                OTc = self.OT_tilde
                Xc = self.X
            else:
                OTc = self.OT_tilde[self.tab_idx[classe_it]]
                Xc = self.X[self.tab_idx[classe_it]]
            if self.metric == "l2":
                C2 = torch.matmul(OTc.sum(0), self.Ysquared.transpose(0, 1)).sum(-1)
                C31 = torch.matmul(Xc.T, OTc)
                C32 = torch.tensordot(C31, self.Y, dims=([0, 2], [2, 0]))
                res = self.Xsquared_sum[classe_it] + C2[None, :] - 2 * C32
            elif self.metric == "l1":
                w1, w2 = self.extended_OT_mat(classe_it)
                C1 = torch.matmul(Xc.transpose(0, -1), w1.T)
                C2 = torch.matmul(self.Y.transpose(0, -1), w2.T)
                C3 = torch.absolute(C1[:, :, None] - C2[:, None, :])
                res = C3.sum(0).sum(-1)
        res /= (self.shapeX[1] + self.shapeY[1]) / 2
        return res

    def get_warp_matrices(self, cl):

        num_path = int(self.pi_DTW_idx[cl].sum())
        Mx = torch.zeros((num_path, self.shapeX[1]))
        My = torch.zeros((num_path, self.shapeY[1]))

        k = 0
        l = 0
        for j in range(0, num_path):
            Mx[j, k] = 1
            My[j, l] = 1
            if (k == self.shapeX[1] - 1) & (l != self.shapeY[1] - 1):
                arg1 = -1
                arg2 = self.pi_DTW_idx[cl][k, l + 1]
                arg3 = -1
            if (l == self.shapeY[1] - 1) & (k != self.shapeX[1] - 1):
                arg1 = self.pi_DTW_idx[cl][k + 1, l]
                arg2 = -1
                arg3 = -1
            if (l != self.shapeY[1] - 1) & (k != self.shapeX[1] - 1):
                arg1 = self.pi_DTW_idx[cl][k + 1, l]
                arg2 = self.pi_DTW_idx[cl][k, l + 1]
                arg3 = self.pi_DTW_idx[cl][k + 1, l + 1]

            pos_move = np.argmax((arg1, arg2, arg3))
            if pos_move == 0:
                k = k + 1
            if pos_move == 1:
                l = l + 1
            if pos_move == 2:
                l = l + 1
                k = k + 1
        return Mx, My

    def path2mat(self, path):
        pi_DTW = torch.zeros((self.shapeX[1], self.shapeY[1]))
        for i, j in path:
            pi_DTW[i, j] = 1
        return pi_DTW

    def extended_OT_mat(self, classe=None):

        if classe is None:
            lenx = self.shapeX[0]
            OTc = self.OT_tilde
        else:
            lenx = len(self.tab_idx[classe])
            OTc = self.OT_tilde[self.tab_idx[classe]]
        el_t = torch.count_nonzero(OTc)

        Wx = torch.zeros(size=(el_t, lenx))
        Wy = torch.zeros(size=(el_t, self.shapeY[0]))
        index_counting_y = 0
        index_counting_x = 0
        for i in range(0, lenx):
            cnt_z = torch.count_nonzero(OTc[i, :])
            nnz = torch.nonzero(OTc[i, :])
            Wx[index_counting_x: index_counting_x + cnt_z, i] = OTc[i, nnz]
            index_counting_x = index_counting_x + cnt_z
            for j in nnz[0]:
                Wy[index_counting_y, j] = OTc[i, j]
                index_counting_y = index_counting_y + 1
        return Wx, Wy

    def stopping_criterion(self, last_pi_DTW):
        stop = True
        for cl in self.classe_unique:
            pi_DTW = self.pi_DTW_idx[cl]
            last_DTW = last_pi_DTW[cl]
            if (pi_DTW != last_DTW).any():
                stop = False
        return stop

    def main_training(self, max_init=100, first_step_DTW=True, un_reg=0.1, un_mas=1.0, transport="emd"):
        ct_cost_OT = 0
        ct_OT = 0
        ct_cost_DTW = 0
        ct_DTW = 0
        cost = {"Cost": []}
        stop = False
        current_init = 0
        # Begin training
        prop_show = []
        for cl in range(0, len(torch.unique(self.classe))):
            prop_show.append(round(torch.sum(self.Xa_one[self.classe == cl]).item(), 4))
        while stop is not True and current_init < max_init:
            if (current_init != 0) or (first_step_DTW is False):
                t_cost_ot = time.time()
                Cost_OT_alpha = self.alpha * self.mat_cost_OT()
                t11 = time.time()
                if self.beta != 0:
                    Cost_0T_beta = self.beta * self.add_cost
                    Cost_OT = Cost_OT_alpha + Cost_0T_beta
                else:
                    Cost_OT = Cost_OT_alpha
                Cost_OT = Cost_OT
                t_ot = time.time()
                ct_cost_OT += t_ot - t_cost_ot
                if transport == "mm_unbalanced":
                    #  OT_tilde = ot.unbalanced.sinkhorn_knopp_unbalanced(self.Xa_one.numpy(), self.Ya_one.numpy(), Cost_OT.numpy(), reg=un_reg, reg_m=un_mas)
                    OT_tilde = ot.unbalanced.mm_unbalanced(self.Xa_one.numpy(), self.Ya_one.numpy(), Cost_OT.numpy(), reg_m=un_mas, div=un_reg)
                    OT_tilde = torch.from_numpy(OT_tilde).type(torch.float)
                elif transport == "partial":
                    OT_tilde = ot.partial.partial_wasserstein(self.Xa_one.numpy(), self.Ya_one.numpy(), Cost_OT.numpy(), m=un_mas, nb_dummies=4)
                    OT_tilde = torch.from_numpy(OT_tilde).type(torch.float)
                elif transport == "emd":
                    OT_tilde = ot.emd(self.Xa_one.cpu(), self.Ya_one.cpu(), Cost_OT.cpu())
                if (torch.cuda.is_available()) and self.GPU:
                    OT_tilde = OT_tilde.cuda()
                self.OT_tilde = OT_tilde
                t_after_ot = time.time()
                ct_OT += t_after_ot - t_ot
                score_OT = torch.sum(self.OT_tilde * Cost_OT)
                cost["Cost"].append(score_OT)

            dtw_score = 0
            self.pi_DTW_path_idx = []
            tot_DTW_cost = 0
            total_cost_dtw = []
            #  if current_init != 0:
            for cl in self.classe_unique:
                t_before_DTW_cost = time.time()
                mat_dist = self.mat_dist_DTW(cl)
                tot_DTW_cost += (mat_dist * self.pi_DTW_idx[cl]).sum()
                total_cost_dtw.append(mat_dist)
                t_after_DTW_cost = time.time()
                self.fst_half += (t_after_DTW_cost - t_before_DTW_cost)
                ct_cost_DTW += t_after_DTW_cost - t_before_DTW_cost
                Pi_DTW_path, dtw_score_prov = torch_dtw_CPU(mat_dist)
                t_after_DTW = time.time()
                ct_DTW += t_after_DTW - t_after_DTW_cost
                self.pi_DTW_path_idx.append(Pi_DTW_path)
                Pi_DTW_prov = self.path2mat(Pi_DTW_path)
                if (torch.cuda.is_available()) and self.GPU:
                    Pi_DTW_prov = Pi_DTW_prov.cuda()
                self.pi_DTW_idx[cl] = Pi_DTW_prov
                dtw_score += dtw_score_prov
            cost["Cost"].append(dtw_score)
            if current_init != 0:
                stop = self.stopping_criterion(last_pi_DTW)
            last_pi_DTW = self.pi_DTW_idx.copy()
            current_init = current_init + 1
        else:
            sum_OT = self.OT_tilde.sum()
            return self.OT_tilde, self.pi_DTW_idx, Cost_OT, score_OT

    def to_onehot(self, y=None):
        if y is None:
            y = self.classe.numpy().astype(int)
        n_values = np.max(y) + 1
        return np.eye(n_values)[y]

    def evaluate(self, train_target_label):
        yt_onehot = self.to_onehot()
        y_pred = np.argmax(np.dot(self.OT_tilde.T, yt_onehot), axis=1)
        accuracy = np.mean(y_pred == train_target_label.numpy().astype(int))
        return accuracy, y_pred


class OTDTW_diag:
    def __init__(self, X, Y, classe=None, weights_X=None, weights_Y=None, metric="l2", settings=0, classe_unique=None,
                 previous_DTW=None, additional_cost=None, alpha=1.0, beta=1.0, GPU=False):
        self.GPU = GPU
        if self.GPU:
            self.X = X
            self.Y = Y
        elif self.GPU is False:
            self.X = X.cpu().detach()
            self.Y = Y.cpu().detach()
        self.alpha = alpha
        self.beta = beta
        self.shapeX = X.shape
        self.shapeY = Y.shape
        self.fst_half = 0
        self.fst_half_iter = 0
        if classe is not None:
            classe = classe.type(torch.int32)
            cl_count = 0
            classe_corrected = torch.empty(size=(self.shapeX[0],), dtype=torch.int)
            for cl in torch.unique(classe):
                classe_corrected[classe == cl] = cl_count
                cl_count = cl_count + 1
            self.classe = classe_corrected
            if classe_unique is not None:
                self.classe_unique = classe_unique
            else:
                self.classe_unique = torch.unique(self.classe)
        else:
            self.classe = torch.zeros(size=(self.shapeX[0],), dtype=torch.int)
            self.classe_unique = torch.unique(self.classe)

        if self.shapeX[-1] == 1:
            self.one_dim = True
        else:
            self.one_dim = False
        if weights_X is None:
            self.Xa_one = torch.ones(size=(self.shapeX[0],)) / self.shapeX[0]
        else:
            self.Xa_one = weights_X
        if weights_Y is None:
            self.Ya_one = torch.ones(size=(self.shapeY[0],)) / self.shapeY[0]
        else:
            self.Ya_one = weights_Y

        self.OT_tilde = self.init_OT_matrix(settings)
        self.metric = metric
        self.add_cost = additional_cost
        if (self.add_cost is not None) & (self.GPU is False):
            self.add_cost = self.add_cost.cpu().detach()
        self.tab_idx = []
        self.dist_OT = []
        self.pi_DTW_idx = []
        self.pi_DTW_path_idx = []
        if (torch.cuda.is_available()) and self.GPU:
            self.Xa_one = self.Xa_one.cuda()
            self.Ya_one = self.Ya_one.cuda()
        if self.metric == 'l2':
            self.Xsquared = []
            self.Xsquared_sum = []
            if self.one_dim:
                self.Ysquared = torch.square(self.Y[:, :, 0]).squeeze()
            else:
                self.Ysquared = torch.square(self.Y).squeeze()
        for cl in self.classe_unique:
            if cl == 0:
                self.tab_idx.append((self.classe == 0).nonzero().squeeze())
            else:
                self.tab_idx.append(self.classe.eq(cl).nonzero().squeeze())
            if (torch.cuda.is_available()) and self.GPU:
                self.tab_idx[cl] = self.tab_idx[cl].cuda()
            if previous_DTW is None:
                self.pi_DTW_idx.append(self.init_DTW_matrix(settings))
            else:
                if (torch.cuda.is_available()) and self.GPU:
                    previous_DTW[cl] = previous_DTW[cl].cuda()
                self.pi_DTW_idx.append(previous_DTW[cl])
            if self.metric == 'l2':
                if self.one_dim:
                    X2 = torch.square(self.X[self.tab_idx[cl], :, 0])
                    X2_sum = torch.matmul(self.Xa_one[self.tab_idx[cl]], X2)
                else:
                    X2 = torch.square(self.X[self.tab_idx[cl]])
                    X2_sum = torch.matmul(self.Xa_one[self.tab_idx[cl]], X2.transpose(0, 1)).sum(-1)
                self.Xsquared.append(X2)
                self.Xsquared_sum.append(X2_sum[:, None])

    # Initialisation of the OT_tilde matrix

    def init_OT_matrix(self, settings):
        torch.manual_seed(settings)
        # cost_OT = torch.square(torch.rand(size=(self.shapeX[0], self.shapeY[0]))).squeeze()
        cost_OT = torch.ones(size=(self.shapeX[0], self.shapeY[0]))
        OT_tilde = ot.emd(self.Xa_one, self.Ya_one, cost_OT, numItermax=10000000)
        # OT_tilde = torch.from_numpy(OT_tilde)
        return OT_tilde

    def init_DTW_matrix(self, settings):
        torch.manual_seed(settings)
        DTW_matrix = torch.zeros(size=(self.shapeX[1], self.shapeY[1]))
        ts = [0, 0]
        indices_table = [[1, 0], [0, 1], [1, 1]]
        diag_fixer = 0
        while (ts[0] != self.shapeX[1] - 1) or (ts[1] != self.shapeY[1] - 1):
            DTW_matrix[ts[0], ts[1]] = 1
            if ts[0] == self.shapeX[1] - 1:
                indice_moving = 1
            elif ts[1] == self.shapeY[1] - 1:
                indice_moving = 0
            else:
                if diag_fixer == 4:
                    indice_moving = 0
                    diag_fixer = 0
                else:
                    indice_moving = 2
                    diag_fixer += 1
            ts[0] = ts[0] + indices_table[indice_moving][0]
            ts[1] = ts[1] + indices_table[indice_moving][1]
        DTW_matrix[-1, -1] = 1
        if (torch.cuda.is_available()) and self.GPU:
            DTW_matrix = DTW_matrix.cuda()
        return DTW_matrix

    def mat_cost_OT(self):

        mat_cost = torch.zeros(size=(self.shapeX[0], self.shapeY[0]))
        if (torch.cuda.is_available()) and self.GPU:
            mat_cost = mat_cost.cuda()
        if self.one_dim:
            for cl in self.classe_unique:
                if self.metric == "l2":
                    pi_DTW = self.pi_DTW_idx[cl]
                    C1 = torch.matmul(self.Xsquared[cl], torch.sum(pi_DTW, dim=1))
                    C2 = torch.matmul(self.Ysquared, torch.sum(pi_DTW.T, dim=1))
                    C3 = torch.matmul(torch.matmul(self.X[self.tab_idx[cl], :, 0], pi_DTW[:]), self.Y[:, :, 0].T)
                    res = C1[:, None] + C2[None, :] - 2 * C3

                elif self.metric == "l1":
                    m1, m2 = self.get_warp_matrices(cl)
                    C1 = torch.matmul(self.X[self.tab_idx[cl], :, 0], m1.T)
                    C2 = torch.matmul(self.Y[:, :, 0], m2.T)
                    C3 = C1[:, None, :] - C2[None, :, :]
                    res = torch.absolute(C3).sum(-1)
                mat_cost[self.tab_idx[cl]] = res
        else:
            for cl in self.classe_unique:
                if self.metric == "l2":
                    pi_DTW = self.pi_DTW_idx[cl]
                    C1 = torch.matmul(self.Xsquared[cl].transpose(1, -1), torch.sum(pi_DTW, dim=1)).sum(-1)
                    C2 = torch.matmul(self.Ysquared.transpose(1, -1), torch.sum(pi_DTW.T, dim=1)).sum(-1)
                    C3 = torch.tensordot(torch.matmul(self.X[self.tab_idx[cl]].transpose(1, -1), pi_DTW), self.Y,
                                         dims=([1, 2], [2, 1]))
                    res = C1[:, None] + C2[None, :] - 2 * C3
                elif self.metric == "l1":
                    m1, m2 = self.get_warp_matrices(cl)
                    C1 = torch.matmul(self.X[self.tab_idx[cl]].transpose(1, -1), m1.T)
                    C2 = torch.matmul(self.Y.transpose(1, -1), m2.T)
                    C3 = C1[:, None, :] - C2[None, :, :]
                    res = torch.absolute(C3).sum(-1).sum(-1)
                mat_cost[self.tab_idx[cl]] = res
        mat_cost /= (self.shapeX[1] + self.shapeY[1]) / 2
        return mat_cost

    def mat_dist_DTW(self, classe_it=None):
        if self.one_dim:
            if classe_it is None:
                OTc = self.OT_tilde
                Xc = self.X[:, :, 0]
            else:
                OTc = self.OT_tilde[self.tab_idx[classe_it]]
                Xc = self.X[self.tab_idx[classe_it], :, 0]
            if self.metric == "l2":
                C2 = torch.matmul(OTc.sum(axis=0), self.Ysquared)
                C3 = torch.matmul(torch.matmul(Xc.T, OTc), self.Y[:, :, 0])
                res = self.Xsquared_sum[classe_it] + C2[None, :] - 2 * C3
            elif self.metric == "l1":
                w1, w2 = self.extended_OT_mat(classe_it)
                C1 = torch.matmul(w1, Xc)
                C2 = torch.matmul(w2, self.Y[:, :, 0])
                C3 = torch.absolute(C1[:, :, None] - C2[:, None, :])
                res = C3.sum(0)
        else:
            if classe_it is None:
                OTc = self.OT_tilde
                Xc = self.X
            else:
                OTc = self.OT_tilde[self.tab_idx[classe_it]]
                Xc = self.X[self.tab_idx[classe_it]]
            if self.metric == "l2":
                C2 = torch.matmul(OTc.sum(0), self.Ysquared.transpose(0, 1)).sum(-1)
                C31 = torch.matmul(Xc.T, OTc)
                C32 = torch.tensordot(C31, self.Y, dims=([0, 2], [2, 0]))
                res = self.Xsquared_sum[classe_it] + C2[None, :] - 2 * C32
            elif self.metric == "l1":
                w1, w2 = self.extended_OT_mat(classe_it)
                C1 = torch.matmul(Xc.transpose(0, -1), w1.T)
                C2 = torch.matmul(self.Y.transpose(0, -1), w2.T)
                C3 = torch.absolute(C1[:, :, None] - C2[:, None, :])
                res = C3.sum(0).sum(-1)
        res /= (self.shapeX[1] + self.shapeY[1]) / 2
        return res

    def get_warp_matrices(self, cl):

        num_path = int(self.pi_DTW_idx[cl].sum())
        Mx = torch.zeros((num_path, self.shapeX[1]))
        My = torch.zeros((num_path, self.shapeY[1]))

        k = 0
        l = 0
        for j in range(0, num_path):
            Mx[j, k] = 1
            My[j, l] = 1
            if (k == self.shapeX[1] - 1) & (l != self.shapeY[1] - 1):
                arg1 = -1
                arg2 = self.pi_DTW_idx[cl][k, l + 1]
                arg3 = -1
            if (l == self.shapeY[1] - 1) & (k != self.shapeX[1] - 1):
                arg1 = self.pi_DTW_idx[cl][k + 1, l]
                arg2 = -1
                arg3 = -1
            if (l != self.shapeY[1] - 1) & (k != self.shapeX[1] - 1):
                arg1 = self.pi_DTW_idx[cl][k + 1, l]
                arg2 = self.pi_DTW_idx[cl][k, l + 1]
                arg3 = self.pi_DTW_idx[cl][k + 1, l + 1]

            pos_move = np.argmax((arg1, arg2, arg3))
            if pos_move == 0:
                k = k + 1
            if pos_move == 1:
                l = l + 1
            if pos_move == 2:
                l = l + 1
                k = k + 1
        return Mx, My

    def path2mat(self, path):
        pi_DTW = torch.zeros((self.shapeX[1], self.shapeY[1]))
        for i, j in path:
            pi_DTW[i, j] = 1
        return pi_DTW

    def extended_OT_mat(self, classe=None):

        if classe is None:
            lenx = self.shapeX[0]
            OTc = self.OT_tilde
        else:
            lenx = len(self.tab_idx[classe])
            OTc = self.OT_tilde[self.tab_idx[classe]]
        el_t = torch.count_nonzero(OTc)

        Wx = torch.zeros(size=(el_t, lenx))
        Wy = torch.zeros(size=(el_t, self.shapeY[0]))
        index_counting_y = 0
        index_counting_x = 0
        for i in range(0, lenx):
            cnt_z = torch.count_nonzero(OTc[i, :])
            nnz = torch.nonzero(OTc[i, :])
            Wx[index_counting_x: index_counting_x + cnt_z, i] = OTc[i, nnz]
            index_counting_x = index_counting_x + cnt_z
            for j in nnz[0]:
                Wy[index_counting_y, j] = OTc[i, j]
                index_counting_y = index_counting_y + 1
        return Wx, Wy

    def stopping_criterion(self, last_pi_DTW):
        stop = True
        for cl in self.classe_unique:
            pi_DTW = self.pi_DTW_idx[cl]
            last_DTW = last_pi_DTW[cl]
            if (pi_DTW != last_DTW).any():
                stop = False
        return stop

    def main_training(self, max_init=100, first_step_DTW=True, un_reg=0.1, un_mas=1.0):
        ct_cost_OT = 0
        ct_OT = 0
        cost = {"Cost": []}
        stop = False
        current_init = 0
        # Begin training
        while stop is not True and current_init < max_init:
            if (current_init != 0) or (first_step_DTW is False):
                t_cost_ot = time.time()
                Cost_OT_alpha = self.alpha * self.mat_cost_OT()
                t11 = time.time()
                if self.beta != 0:
                    Cost_0T_beta = self.beta * self.add_cost
                    Cost_OT = Cost_OT_alpha + Cost_0T_beta
                else:
                    Cost_OT = Cost_OT_alpha
                t_ot = time.time()
                ct_cost_OT += t_ot - t_cost_ot
                #  OT_tilde = ot.emd(self.Xa_one.cpu(), self.Ya_one.cpu(), Cost_OT.cpu(), numItermax=1000000)
                OT_tilde = ot.unbalanced.sinkhorn_knopp_unbalanced(self.Xa_one.cpu(), self.Ya_one.cpu(), Cost_OT.cpu(), reg=un_reg, reg_m=un_mas)
                OT_tilde = torch.from_numpy(OT_tilde).type(torch.float)
                # OT_tilde = torch.from_numpy(OT_tilde)
                if (torch.cuda.is_available()) and self.GPU:
                    OT_tilde = OT_tilde.cuda()
                self.OT_tilde = OT_tilde
                t_after_ot = time.time()
                ct_OT += t_after_ot - t_ot
                score_OT = torch.sum(self.OT_tilde * Cost_OT)
                cost["Cost"].append(score_OT)

            if current_init != 0:
                stop = self.stopping_criterion(last_pi_DTW)
            last_pi_DTW = self.pi_DTW_idx.copy()
            current_init = current_init + 1
        else:
            return self.OT_tilde, self.pi_DTW_idx, Cost_OT, score_OT

    def to_onehot(self, y=None):
        if y is None:
            y = self.classe.numpy().astype(int)
        n_values = np.max(y) + 1
        return np.eye(n_values)[y]

    def evaluate(self, train_target_label):
        yt_onehot = self.to_onehot()
        y_pred = np.argmax(np.dot(self.OT.T, yt_onehot), axis=1)
        accuracy = np.mean(y_pred == train_target_label.numpy().astype(int))
        return accuracy, y_pred


@torch.jit.script
def torch_acc_matrix(cost_matrix):
    l1 = cost_matrix.shape[0]
    l2 = cost_matrix.shape[1]
    cum_sum = torch.full((l1 + 1, l2 + 1), torch.inf, dtype=cost_matrix.dtype, device=cost_matrix.device)
    cum_sum[0, 0] = 0.
    cum_sum[1:, 1:] = cost_matrix

    for i in range(l1):
        for j in range(l2):
            cum_sum[i + 1, j + 1] += torch.min(cum_sum[[i, i + 1, i], [j + 1, j, j]])
    return cum_sum[1:, 1:]


@torch.jit.script
def _return_path(acc_cost_mat):
    sz1, sz2 = acc_cost_mat.shape
    path = [(sz1 - 1, sz2 - 1)]
    while path[-1] != (0, 0):
        i, j = path[-1]
        if i == 0:
            path.append((0, j - 1))
        elif j == 0:
            path.append((i - 1, 0))
        else:
            arr = acc_cost_mat[[i - 1, i - 1,  i], [j - 1, j, j - 1]]
            argmin = torch.argmin(arr)
            if argmin == 0:
                path.append((i - 1, j - 1))
            elif argmin == 1:
                path.append((i - 1, j))
            else:
                path.append((i, j - 1))
    return path[::-1]


def torch_dtw(cost_matrix):
    acc_matrix = torch_acc_matrix(cost_matrix=cost_matrix)
    path = _return_path(acc_matrix)
    return path, acc_matrix[-1, -1]


@torch.jit.script
def torch_acc_matrix_CPU(cost_matrix):
    l1 = cost_matrix.shape[0]
    l2 = cost_matrix.shape[1]
    cum_sum = torch.full((l1 + 1, l2 + 1), torch.inf, dtype=cost_matrix.dtype)
    cum_sum[0, 0] = 0.
    cum_sum[1:, 1:] = cost_matrix

    for i in range(l1):
        for j in range(l2):
            cum_sum[i + 1, j + 1] += torch.min(cum_sum[[i, i + 1, i], [j + 1, j, j]])
    return cum_sum[1:, 1:]


@torch.jit.script
def _return_path_CPU(acc_cost_mat):
    sz1, sz2 = acc_cost_mat.shape
    path = [(sz1 - 1, sz2 - 1)]
    while path[-1] != (0, 0):
        i, j = path[-1]
        if i == 0:
            path.append((0, j - 1))
        elif j == 0:
            path.append((i - 1, 0))
        else:
            arr = acc_cost_mat[[i - 1, i - 1,  i], [j - 1, j, j - 1]]
            argmin = torch.argmin(arr)
            if argmin == 0:
                path.append((i - 1, j - 1))
            elif argmin == 1:
                path.append((i - 1, j))
            else:
                path.append((i, j - 1))
    return path[::-1]


def torch_dtw_CPU(cost_matrix):
    acc_matrix = torch_acc_matrix_CPU(cost_matrix=cost_matrix)
    path = _return_path_CPU(acc_matrix)
    return path, acc_matrix[-1, -1]


if __name__ == "__main__":

    def from_numpy_to_torch(filename, float_or_long=True):
        data = np.load(filename)
        data_t = torch.from_numpy(data)
        if float_or_long:
            data_t = data_t.type(torch.float)
        else:
            data_t = data_t.type(torch.long)
        return data_t

    def to_onehot(y):
        n_values = np.max(y) + 1
        return np.eye(n_values)[y]

    def test_unbalanced(data_name, domain_list, reg_list, mas_list, transport):
        path = "/home/adr2.local/painblanc_f/codats-master/datas/numpy_data/" + data_name
        train_source = from_numpy_to_torch(path + str(domain_list[0]) + "test.npy")
        train_source_label = from_numpy_to_torch(path + str(domain_list[0]) + "test_labels.npy")
        train_target = from_numpy_to_torch(path + str(domain_list[1]) + "test.npy")
        train_target_label = from_numpy_to_torch(path + str(domain_list[1]) + "test_labels.npy")
        sum_mat = np.empty(shape=(len(reg_list), len(mas_list), 2))
        row_i = 0
        for r in reg_list:
            col_i = 0
            for m in mas_list:
                for t in transport:
                    #  mad = OTDTW_CPU_UNBALANCED(train_source, train_target, beta=0.0)
                    #  mad = OTDTW_CPU_UNBALANCED(train_source, train_target, beta=0.0, metric="l2")
                    #  mad = OTDTW_diag(train_source, train_target, train_source_label, beta=0.0)
                    mad = OTDTW_CPU_UNBALANCED(train_source, train_target, train_source_label, beta=0.0)
                    OT, DTW, Cost_OT, score_OT = mad.main_training(un_reg=r, un_mas=m, transport=t)
                    #  la_tante_rep(OT=OT, DTW=DTW, OT_cost=Cost_OT, source=train_source[:100], target=train_target[:100], source_label=train_source_label[:100])
                    acc, _ = mad.evaluate(train_target_label)
                    sum_mat[row_i, col_i, 0] = mad.OT_tilde.sum()
                    sum_mat[row_i, col_i, 1] = score_OT
                col_i += 1
            row_i += 1
        return sum_mat

    def la_tante_rep(OT, DTW, OT_cost, source, target, source_label):
        OT = OT.numpy()
        from sklearn.metrics.pairwise import euclidean_distances
        source_dist = euclidean_distances(source.reshape(source.shape[0], -1), squared=True)
        source_dist /= source.shape[1]
        target_dist = euclidean_distances(target.reshape(target.shape[0], -1), squared=True)
        target_dist /= target.shape[1]
        source_target_cost_w = np.empty(shape=(source.shape[0], target.shape[0]))
        c_dtw = 0
        for c in range(0, len(torch.unique(source_label))):
            c_dtw = c
            source_target_cost_w[source_label == c] = (OT_cost[source_label == c] * source.shape[2]) / DTW[c_dtw].sum()

        max_OT = OT * (OT >= np.sort(OT, axis=1)[:, [-1]])
        max_OT += OT * (OT >= np.sort(OT, axis=1)[:, [-2]])
        max_OT += OT * (OT >= np.sort(OT, axis=1)[:, [-3]])
        indice_matrix = np.nonzero(max_OT)
        ind_row = indice_matrix[0]
        ind_col = indice_matrix[1]
        Cost1 = np.concatenate((source_dist, source_target_cost_w), axis=1)
        Cost2 = np.concatenate((source_target_cost_w.T, target_dist), axis=1)
        Cost = np.concatenate((Cost1, Cost2), axis=0)
        transform = MDS(n_components=2, random_state=1, dissimilarity="precomputed").fit_transform(Cost)
        source_transform = transform[:source.shape[0]]
        target_transform = transform[source.shape[0]:]
        for pair in range(0, len(ind_row)):
            if pair == 0:
                plt.plot([source_transform[ind_row[pair], 0], target_transform[ind_col[pair], 0]],
                            [source_transform[ind_row[pair], 1], target_transform[ind_col[pair], 1]],
                            color="black", linestyle='--', linewidth=0.3)
            else:
                plt.plot([source_transform[ind_row[pair], 0], target_transform[ind_col[pair], 0]],
                            [source_transform[ind_row[pair], 1], target_transform[ind_col[pair], 1]],
                            color="black", linestyle='--', linewidth=0.3)
        plt.title("Projection")
        plt.scatter(source_transform[:, 0],
                    source_transform[:, 1],
                    marker='.', label="source MAD", s=100)
        plt.scatter(target_transform[:, 0],
                    target_transform[:, 1],
                    marker='s', label="target MAD", s=20)
        plt.legend()
        plt.show()

    plt.rc("font", size=15)

    source = np.load("trace_source_same.npy")
    target = np.load("trace_target_same.npy")
    source_label = np.zeros(shape=(source.shape[0]))
    target_label = np.zeros(shape=(target.shape[0]))
    source_label[20:] = 1
    target_label[10:] = 1

    weight_X = np.empty(shape=(30))
    weight_X[source_label == 0] = 1 / (3 * 20)
    weight_X[source_label == 1] = 2 / (3 * 10)
    print(np.sum(weight_X))
    weight_X=None
    mad = OTDTW_CPU_UNBALANCED(torch.Tensor(source), torch.Tensor(target), torch.Tensor(source_label), weights_X=weight_X, beta=0.0)
    OT, DTW, Cost_OT, score_OT = mad.main_training(un_reg="kl", un_mas=0.70, transport="emd")
    acc, _ = mad.evaluate(torch.Tensor(target_label))
    print(acc)
    print(OT.sum())
    plt.imshow(OT)
    plt.show()
    """sum_mat = test_unbalanced("ucihar_", domain_list=[12, 16], reg_list=["l2"], mas_list=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0])
    plt.plot(sum_mat[0], label="12 16", linewidth=5)
    sum_mat = test_unbalanced("ucihar_", domain_list=[12, 18], reg_list=["l2"], mas_list=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0])
    plt.plot(sum_mat[0], label="12 18", linewidth=5)
    sum_mat = test_unbalanced("ucihar_", domain_list=[14, 19], reg_list=["l2"], mas_list=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0])
    plt.plot(sum_mat[0], label="14 19", linewidth=5)"""
    
    """sum_mat = test_unbalanced("ucihar_", domain_list=[12, 16], reg_list=[0.1], mas_list=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0])
    plt.plot(sum_mat[0], label="12 16", linewidth=5)
    sum_mat = test_unbalanced("ucihar_", domain_list=[12, 18], reg_list=[0.01], mas_list=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0])
    plt.plot(sum_mat[0], label="12 18", linewidth=5)
    sum_mat = test_unbalanced("ucihar_", domain_list=[14, 19], reg_list=[0.01], mas_list=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0])
    plt.plot(sum_mat[0], label="14 19", linewidth=5)"""
    """sum_mat = test_unbalanced("remotes7cl_", domain_list=[3, 1], reg_list=["kl"], mas_list=[1.0, 5.0, 10.0, 50.0, 100.0, 500.0], transport=["mm_unbalanced"])
    sum_mat_tot.append(sum_mat)
    sum_mat = test_unbalanced("remotes7cl_", domain_list=[3, 2], reg_list=["kl"], mas_list=[1.0, 5.0, 10.0, 50.0, 100.0, 500.0], transport=["mm_unbalanced"])
    sum_mat_tot.append(sum_mat)
    sum_mat = test_unbalanced("remotes7cl_", domain_list=[3, 4], reg_list=["kl"], mas_list=[1.0, 5.0, 10.0, 50.0, 100.0, 500.0], transport=["mm_unbalanced"])
    sum_mat_tot.append(sum_mat)
    sum_mat = test_unbalanced("remotes7cl_", domain_list=[1, 2], reg_list=["kl"], mas_list=[1.0, 5.0, 10.0, 50.0, 100.0, 500.0], transport=["mm_unbalanced"])
    sum_mat_tot.append(sum_mat)
    sum_mat = test_unbalanced("remotes7cl_", domain_list=[1, 3], reg_list=["kl"], mas_list=[1.0, 5.0, 10.0, 50.0, 100.0, 500.0], transport=["mm_unbalanced"])
    sum_mat_tot.append(sum_mat)
    np.save('total_score_unbal.npy', sum_mat_tot)"""
    
    """sum_mat_tot = []
    sum_mat = test_unbalanced("remotes7cl_", domain_list=[3, 1], reg_list=["kl"], mas_list=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], transport=["partial"])
    sum_mat_tot.append(sum_mat)
    sum_mat = test_unbalanced("remotes7cl_", domain_list=[3, 2], reg_list=["kl"], mas_list=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], transport=["partial"])
    sum_mat_tot.append(sum_mat)
    sum_mat = test_unbalanced("remotes7cl_", domain_list=[3, 4], reg_list=["kl"], mas_list=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], transport=["partial"])
    sum_mat_tot.append(sum_mat)
    sum_mat = test_unbalanced("remotes7cl_", domain_list=[1, 2], reg_list=["kl"], mas_list=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], transport=["partial"])
    sum_mat_tot.append(sum_mat)
    sum_mat = test_unbalanced("remotes7cl_", domain_list=[1, 3], reg_list=["kl"], mas_list=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], transport=["partial"])
    sum_mat_tot.append(sum_mat)
    np.save('total_score_partial.npy', sum_mat_tot)"""

    """sum_mat = test_unbalanced("remotes7cl_", domain_list=[3, 2], reg_list=["kl"], mas_list=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0])
    plt.plot(sum_mat[0], label="kl", linewidth=5)"""
    """sum_mat = test_unbalanced("remotes7cl_", domain_list=[3, 1], reg_list=["l2"], mas_list=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0])
    plt.plot(sum_mat[0], label="Dan2Bzh", linewidth=5)
    sum_mat = test_unbalanced("remotes7cl_", domain_list=[3, 4], reg_list=["l2"], mas_list=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0])
    plt.plot(sum_mat[0], label="Dan2Aus", linewidth=5)
    sum_mat = test_unbalanced("remotes7cl_", domain_list=[1, 2], reg_list=["l2"], mas_list=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0])
    plt.plot(sum_mat[0], label="Bzh2Tarn", linewidth=5)
    sum_mat = test_unbalanced("remotes7cl_", domain_list=[1, 3], reg_list=["l2"], mas_list=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0])
    plt.plot(sum_mat[0], label="Bzh2Dan", linewidth=5)"""
    """plt.plot([0.0, 8.0], [1.0, 1.0], color="black", linewidth=5)
    plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8], ["0.01", "0.05", "0.1", "0.5", "1", "5", "10", "50", "100"])
    plt.legend(title="Adaptation problem")
    plt.xlabel("Mass regularisation term")
    plt.ylabel("Sum of tranported mass")
    plt.title("MM Unbalanced MAD")
    plt.show()
 
    path = "/home/adr2.local/painblanc_f/codats-master/datas/numpy_data/remotes7cl_"

    train_source = from_numpy_to_torch(path + "1train.npy")
    train_source_label = from_numpy_to_torch(path + "1train_labels.npy")
    train_target = from_numpy_to_torch(path + "2train.npy")
    train_target_label = from_numpy_to_torch(path + "2train_labels.npy")

    train_source = from_numpy_to_torch("REMOTES/Bzh2Tarn_unbal3000testConv_source.npy")
    train_source_label = from_numpy_to_torch("REMOTES/Bzh2Tarn_unbal3000testlabels_source.npy")
    train_target = from_numpy_to_torch("REMOTES/Bzh2Tarn_unbal3000testConv_target.npy")
    train_target_label = from_numpy_to_torch("REMOTES/Bzh2Tarn_unbal3000testlabels_target.npy")
    train_source = train_source.transpose(-1, 1)
    train_target = train_target.transpose(-1, 1)
    source = 11
    target = 12
    n_classes = 5
    batchsize = 256"""

    """mad = OTDTW_CPU_UNBALANCED(train_source, train_target, train_source_label, beta=0.0)
    mad.main_training()
    plt.imshow(mad.OT_tilde)
    plt.show()
    acc, _ = mad.evaluate(train_target_label)
    for i in range(len(mad.pi_DTW_idx)):
        plt.clf()
        plt.imshow(mad.pi_DTW_idx[i])
        plt.show()

    sample_vec = torch.zeros(size=(n_classes,))
    _, count_classes = torch.unique(train_source_label, return_counts=True)
    for cl in range(0, n_classes):
        cl_bs = torch.round(batchsize * count_classes[cl] / torch.sum(count_classes))
        if cl_bs <= 1:
            cl_bs += 1
        sample_vec[cl] = cl_bs
    while sample_vec.sum() > batchsize:
        sample_vec[torch.argmax(sample_vec)] -= 1"""

    def mini_batch_class_balanced(X, vector, n_classes, batchsize, y=None, shuffle=True):
        gen = torch.Generator()
        if y is not None:
            if shuffle:
                rindex = torch.randperm(len(X))
                X = X[rindex]
                y = y[rindex]
            index = torch.tensor([])
            for i in range(n_classes):
                s_index = torch.nonzero(y == i).squeeze()
                index_random = torch.randperm(n=s_index.shape[0], generator=gen)
                s_ind = s_index[index_random]
                index = torch.cat((index, s_ind[0:vector[i].item()]), 0)
            index = index.type(torch.long)
            index = index.view(-1)
            index_rand = torch.randperm(len(index), generator=gen)
            index = index[index_rand]
            X_minibatch, y_minibatch = X[index], y[index]
            return X_minibatch.float(), y_minibatch.long(), index
        else:
            index = torch.randperm(len(X), generator=gen)
            index = index[:batchsize]
            X_target_minibatch = torch.tensor(X[index])
            return X_target_minibatch.float(), y, index

    def mat_dist_DTW(X_source, X_target, X_target_squared):
        Xc_squared = X_source ** 2
        res = Xc_squared.sum(-1)[:, :, None, None] + X_target_squared.sum(-1)[None, None, :, :] - 2 * np.dot(X_source, X_target.transpose(0, -1, 1))
        res = res.sum(0).sum(1)
        return res

    def path2mat(path, shapeX, shapeY):
        pi_DTW = np.zeros((shapeX, shapeY))
        for i, j in path:
            pi_DTW[i, j] = 1
        return pi_DTW

    def DTW_classes(X_source, X_target, Y_source):
        X_target_squared = X_target ** 2
        list_dtw = []
        for cl in np.unique(Y_source):
            tab_idx = np.where(Y_source == cl)
            Xc = X_source[tab_idx]
            metric = mat_dist_DTW(Xc, X_target, X_target_squared)
            dtw_path, _ = tslm.dtw_path_from_metric(metric, metric="precomputed")
            dtw = path2mat(dtw_path, 72, 58)
            list_dtw.append(dtw)
        return list_dtw
