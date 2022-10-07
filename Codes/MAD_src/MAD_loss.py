import torch
import numpy as np
from OTDTW import OTDTW_CPU, OTDTW_torch32, OTDTW_diag, OTDTW_CPU_UNBALANCED
import time


class MAD_loss(torch.nn.Module):
    def __init__(self, num_class, CUDA_train, CUDA_MAD, MAD_class, alpha, beta, torching, target_prop=None,
                 unbalanced=False, un_reg=None, un_mas=1.0):
        super().__init__()
        self.num_class = num_class
        self.CUDA_train = CUDA_train
        self.DTW_ = None
        self.CUDA_MAD = CUDA_MAD
        self.MAD_class = MAD_class
        self.alpha = alpha
        self.beta = beta
        self.torching = torching
        self.target_prop = target_prop
        self.unbalanced = unbalanced
        self.un_reg = un_reg
        self.un_mas = un_mas

    def mad(self, out_conv_source, out_conv_target, labels_source, similarity_CE, sample_source=None):
        """

        :param out_conv_source:
        :param out_conv_target:
        :param labels_source:
        :return:
        Examples:
        ---------
        """
        with torch.no_grad():
            if self.MAD_class is not True:
                labels_source = None
                weight_X = None
            elif self.target_prop is not None:
                weight_X = torch.empty(size=labels_source.shape)
                for cl in range(0, len(sample_source)):
                    weight_X[labels_source == cl] = self.target_prop[cl] / (sample_source[cl])
            else:
                weight_X = None
            if self.torching:
                if self.CUDA_train:
                    mad = OTDTW_torch32(out_conv_source.transpose(1, 2),
                                        out_conv_target.transpose(1, 2), labels_source, weights_X=weight_X,
                                        classe_unique=torch.arange(self.num_class), metric="l2",
                                        previous_DTW=self.DTW_,
                                        additional_cost=similarity_CE, alpha=self.alpha, beta=self.beta)
                    self.OT_, self.DTW_, self.cost_OT_, self._score = mad.main_training(first_step_DTW=False)
                elif self.unbalanced:
                    if self.un_reg is None:
                        transport = "partial"
                    else:
                        transport = "mm_unbalanced"
                    mad = OTDTW_CPU_UNBALANCED(out_conv_source.transpose(1, 2),
                                               out_conv_target.transpose(1, 2), labels_source, weights_X=weight_X,
                                               classe_unique=torch.arange(self.num_class), metric="l2",
                                               previous_DTW=self.DTW_, additional_cost=similarity_CE,
                                               alpha=self.alpha, beta=self.beta)
                    self.OT_, self.DTW_, self.cost_OT_, self._score = mad.main_training(first_step_DTW=False, un_reg=self.un_reg, un_mas=self.un_mas, transport=transport)
                else:
                    mad = OTDTW_CPU(out_conv_source.transpose(1, 2),
                                    out_conv_target.transpose(1, 2), labels_source, weights_X=weight_X,
                                    classe_unique=torch.arange(self.num_class), metric="l2",
                                    previous_DTW=self.DTW_,
                                    additional_cost=similarity_CE, alpha=self.alpha, beta=self.beta, GPU=self.CUDA_MAD)
                    self.OT_, self.DTW_, self.cost_OT_, self._score = mad.main_training(first_step_DTW=False)
            else:
                mad = OTDTW_diag(out_conv_source.transpose(1, 2),
                                 out_conv_target.transpose(1, 2), labels_source,
                                 classe_unique=np.arange(self.num_class), metric="l2", previous_DTW=self.DTW_,
                                 additional_cost=similarity_CE, alpha=self.alpha, beta=self.beta)
                self.OT_, self.DTW_, self.cost_OT_, self._score_ = mad.main_training(first_step_DTW=False)

    def l2_torch(self, labels_source, out_conv_source, out_conv_target, loop_iteration, OT):
        if self.CUDA_train:
            global_l2_matrix = torch.zeros(size=(out_conv_source.shape[0], out_conv_target.shape[0]), device="cuda:0")
        else:
            global_l2_matrix = torch.zeros(size=(out_conv_source.shape[0], out_conv_target.shape[0]))
            if (torch.cuda.is_available()) and self.CUDA_MAD:
                global_l2_matrix = global_l2_matrix.cuda()
        out_conv_source_sq = out_conv_source ** 2
        out_conv_target_sq = out_conv_target ** 2
        for cl in range(0, loop_iteration):
            if loop_iteration == 1:
                idx_cl = torch.arange(0, labels_source.shape[0], 1)
            else:
                idx_cl = torch.where(labels_source == cl)

            pi_DTW = self.DTW_[cl]
            pi_DTW = torch.tensor(pi_DTW)
            C1 = torch.matmul(out_conv_source_sq[idx_cl], torch.sum(pi_DTW, dim=1)).sum(-1)
            C2 = torch.matmul(out_conv_target_sq, torch.sum(pi_DTW.T, dim=1)).sum(-1)
            C3 = torch.tensordot(torch.matmul(out_conv_source[idx_cl], pi_DTW), out_conv_target,
                                 dims=([1, 2], [1, 2]))
            C4 = C1[:, None] + C2[None, :] - 2 * C3
            global_l2_matrix[idx_cl] = C4
        l2_OT_matrix = OT * global_l2_matrix
        return l2_OT_matrix

    def forward(self, out_conv_source, out_conv_target, labels_source, similarity_CE, source_sample=None):
        """

        :param out_conv_source:
        :param out_conv_target:
        :param labels_source:
        :return:

        examples:
        ---------
        >>> source = torch.rand(size=(2000, 1, 200))
        >>> target = 10 * torch.rand(size=(2000, 1, 200))
        >>> labels = torch.zeros(size=(2000,))
        >>> mad_test = MAD_loss(num_class=1, MAD_class=True, CUDA_train=False)
        >>> alpha_loss, OT, DTW, cost_OT = mad_test.loss_CNN_MAD(out_conv_source=source, out_conv_target=target, labels_source=labels)
        """
        self.mad(out_conv_source, out_conv_target, labels_source, similarity_CE, source_sample)
        #  self.OT_ = torch.tensor(self.OT_)
        self.DTW_cuda = []
        if self.CUDA_train:
            if torch.cuda.is_available():
                self.OT_ = self.OT_.cuda()
                for cl in range(0, len(self.DTW_)):
                    self.DTW_cuda.append(torch.tensor(self.DTW_[cl]).cuda())
        else:
            self.DTW_cuda = self.DTW_

        if self.MAD_class:
            loop_iteration = torch.max(labels_source).item() + 1
        else:
            loop_iteration = 1
        alpha_loss = self.l2_torch(labels_source=labels_source, out_conv_source=out_conv_source,
                                   loop_iteration=int(loop_iteration), OT=self.OT_, out_conv_target=out_conv_target)

        length = (out_conv_source.shape[-1] + out_conv_target.shape[-1]) / 2
        beta_loss = (self.OT_ * similarity_CE).sum()
        return self.alpha * alpha_loss.sum() / length, self.beta * beta_loss
