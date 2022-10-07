import sklearn.metrics
import torch
import torch.nn as nn
import numpy.random as npr
from MAD_loss import MAD_loss
from torch.utils.data import Dataset
import numpy as np
import warnings


class Data_set(Dataset):
    def __init__(self, X, Y=None):
        self.X = X
        self.Y = Y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, item):
        if self.Y is None:
            return self.X[item]
        else:
            return self.X[item], self.Y[item]


class CNNMAD(nn.Module):
    def __init__(self, name, batchsize, channel,
                 train_source_data, train_source_label,
                 MAD_class=True, reg=0., num_class=6, lr=0.001,
                 train_target_data=None, train_target_label=None,
                 valid_source_data=None, valid_source_label=None, valid_target_data=None, valid_target_label=None,
                 test_source_data=None, test_source_label=None, test_target_data=None, test_target_label=None,
                 big_model=True, saving=True, CUDA_train=False, batchnorm=True, reg_rep=True, affine=True):
        super(CNNMAD, self).__init__()
        torch.manual_seed(200)
        np.random.seed(200)
        self.CUDA_train = CUDA_train
        self.name = name
        self.reg = reg
        self.reg_rep = reg_rep
        self.affine = affine
        self.lr = lr
        self.num_class = num_class
        self.saving = saving
        self.last_accuracy = 0
        self.DTW = None
        self.trainSourceData = train_source_data
        self.trainSourceLabel = train_source_label
        if train_target_data is not None:
            self.trainTargetData = train_target_data
            self.trainTargetLabel = train_target_label
        if valid_target_data is not None:
            self.validTargetData = valid_target_data
            self.validTargetLabel = valid_target_label
            self.validSourceData = valid_source_data
            self.validSourceLabel = valid_source_label
        else:
            self.validTargetData = train_target_data
            self.validTargetLabel = train_target_label
            self.validSourceData = train_source_data
            self.validSourceLabel = train_source_label
        if test_source_data is not None:
            self.testSourceData = test_source_data
            self.testSourceLabel = test_source_label
            self.testTargetData = test_target_data
            self.testTargetLabel = test_target_label
        else:
            self.testSourceData = train_source_data
            self.testSourceLabel = train_source_label
            self.testTargetData = train_target_data
            self.testTargetLabel = train_target_label
        self.batchsize = batchsize
        if big_model:
            if batchnorm:
                self.conv1 = nn.Sequential(nn.Conv1d(in_channels=channel, out_channels=128, kernel_size=8, stride=1,
                                                     padding="same", bias=False),
                                           nn.BatchNorm1d(num_features=128, affine=self.affine),
                                           nn.ReLU())
                self.conv2 = nn.Sequential(nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, stride=1,
                                                     padding="same", bias=False),
                                           nn.BatchNorm1d(num_features=256, affine=self.affine),
                                           nn.ReLU())
                self.conv3 = nn.Sequential(nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, stride=1,
                                                     padding="same", bias=False),
                                           nn.BatchNorm1d(num_features=128, affine=self.affine),
                                           nn.ReLU())
            else:
                self.conv1 = nn.Sequential(nn.Conv1d(in_channels=channel, out_channels=128, kernel_size=8, stride=1,
                                                     padding="same", bias=False),
                                           nn.ReLU())
                self.conv2 = nn.Sequential(nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, stride=1,
                                                     padding="same", bias=False),
                                           nn.ReLU())
                self.conv3 = nn.Sequential(nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, stride=1,
                                                     padding="same", bias=False),
                                           nn.ReLU())
        else:
            self.conv1 = nn.Sequential(nn.Conv1d(in_channels=channel, out_channels=32, kernel_size=8, stride=1,
                                                 padding="same", bias=True),
                                       #  nn.BatchNorm1d(num_features=128),
                                       nn.ReLU())

        torch.nn.init.xavier_uniform_(self.conv1[0].weight)
        torch.nn.init.xavier_uniform_(self.conv2[0].weight)
        torch.nn.init.xavier_uniform_(self.conv3[0].weight)
        self.optimizer = torch.optim.Adam([{'params': self.conv1.parameters()},
                                           {'params': self.conv2.parameters()},
                                           {'params': self.conv3.parameters()}],
                                          lr=lr)

        self.crossLoss = nn.CrossEntropyLoss()
        self.epoch = 0
        self.iteration = 0
        self.best_iteration = 0
        self.best_iteration_cnn = 0
        self.lowest_loss = 1000000
        self.lowest_loss_cnn = 1000000
        self.loss_count = []
        self.loss_count_cnn_valid_source = []
        self.loss_count_valid_source = []
        self.loss_count_valid_target = []
        self.loss_beta = []
        self.loss_alpha = []
        self.acc_source = []
        self.acc_target = []
        self.norm_conv1 = []
        self.weight_con1 = []
        self.sample_size = int(self.batchsize / self.num_class)
        self.MAD_class = MAD_class
        self.best_model = self.state_dict()
        self.best_model_cnn = self.state_dict()
        self.MAD_loss_train = MAD_loss(num_class=self.num_class, CUDA_train=self.CUDA_train, MAD_class=self.MAD_class)
        self.MAD_loss_valid = MAD_loss(num_class=self.num_class, CUDA_train=self.CUDA_train, MAD_class=self.MAD_class)
        if self.CUDA_train:
            if torch.cuda.is_available():
                self.conv1 = self.conv1.cuda()

    """def mad(self, out_conv_source, out_conv_target, labels_source, validation=False):
        
        compute mad, to develop
        parameters :
        out
        examples
        --------
        >>> import numpy.random as npr
        >>> npr.seed(1)
        >>> labels = torch.tensor(npr.random_integers(low=0, high=2, size=10)).type(torch.float)
        >>> source = npr.random(size=(100, 100, 1))
        >>> target = npr.random(size=(100, 100, 1))
        >>> source_conv = torch.zeros(size=(10, 1, 128)).type(torch.float)
        >>> target_conv = torch.zeros(size=(20, 1, 128)).type(torch.float)
        >>> cnnmad = CNNMAD(name="", batchsize=20, channel=1, train_source_data=source, train_source_label=labels, num_class=3, train_target_data=target, train_target_label=labels)
        >>> cnnmad.mad(out_conv_source=source_conv, out_conv_target=target_conv, labels_source=labels)
        >>> print(cnnmad._OT.shape)
        (10, 20)
        >>> print(cnnmad._cost_OT.shape)
        (10, 20)
        >>> print(cnnmad._OT.sum())
        1.0
        >>> print((cnnmad._OT * cnnmad._cost_OT).sum())
        0.0
        
        with torch.no_grad():
            if self.MAD_class is not True:
                labels_source = None
            mad = OTDTW(out_conv_source.type(torch.double).transpose(1, 2), out_conv_target.type(torch.double).transpose(1, 2),
                        labels_source, metric="l2", classe_unique=np.arange(self.num_class), previous_DTW=self.DTW)
            if validation:
                self._OT_valid, self.DTW_valid, self._cost_OT_valid, _ = mad.main_training(first_step_DTW=False)
            else:
                self._OT, self.DTW, self._cost_OT, score = mad.main_training(first_step_DTW=False)"""

    def mini_batch_class_balanced(self, shuffle=True):
        """
        Draw minibatches with the same number of series for each classes
        :parameter
        shuffle : if true the dataset is shuffled before drawing
        :return
        the index of the series to pick
        examples
        --------
        >>> import numpy.random as npr
        >>> npr.seed(1)
        >>> labels = npr.random_integers(low=0, high=2, size=100)
        >>> series = npr.random(size=(100, 100, 1))
        >>> cnnmad = CNNMAD(name="", batchsize=10, channel=1, train_source_data=series, train_source_label=labels, num_class=3)
        >>> index = cnnmad.mini_batch_class_balanced(shuffle=False)
        >>> print(labels[index])
        [0 0 0 1 1 1 2 2 2]
        """
        if shuffle:
            rindex = np.random.permutation(len(self.trainSourceLabel))
            self.trainSourceLabel = self.trainSourceLabel[rindex]
            self.trainSourceData = self.trainSourceData[rindex]

        index = []
        for i in range(self.num_class):
            s_index = np.array(np.nonzero(self.trainSourceLabel == i))
            s_ind = npr.permutation(s_index.reshape(-1))
            index = np.append(index, s_ind[0:self.sample_size])
        index = np.array(index, dtype=int)
        # index = npr.permutation(index)
        return index

    """def l2_torch(self, labels_source, out_conv_source, out_conv_target, loop_iteration, OT):
        

        :param labels:
        :param out_source:
        :param out_target:
        :param loop_iteration:
        :param OT:
        :return:

        examples
        --------
        >>> source = torch.rand(size=(2000, 1, 200))
        >>> source.requires_grad = True
        >>> target = 10 * torch.rand(size=(2000, 1, 200))
        >>> target.requires_grad = True
        >>> labels = torch.zeros(size=(2000,), requires_grad=True)
        >>> cnnmad = CNNMAD(name="", batchsize=10, beta=1.0, alpha=0.1, channel=1, train_source_data=source, train_source_label=labels, num_class=1)
        >>> cnnmad.set_current_batchsize(2000)
        >>> cnnmad.set_current_batchsize(2000, train=False)
        >>> cnnmad.mad(out_conv_source=source, out_conv_target=target, labels_source=labels)
        >>> OT = torch.tensor(cnnmad._OT, requires_grad=True)
        >>> print(OT.sum().item())
        1.0
        >>> out_cnn = cnnmad.l2_torch(labels_source=labels, out_conv_source=source, out_conv_target=target, loop_iteration=1, OT=OT)
        >>> print((cnnmad._OT * cnnmad._cost_OT).sum() - out_cnn.sum().item() < 1e-6)
        True
        >>> out_cnn.sum().backward()
        >>> print(target.grad)
        out_conv_source = out_conv_source.type(torch.double)
        out_conv_target = out_conv_target.type(torch.double)
        out_conv_source_sq = out_conv_source ** 2
        out_conv_target_sq = out_conv_target ** 2

        idx_cl = torch.where(labels_source == 0)
        pi_DTW = self.DTW[0]
        pi_DTW = torch.tensor(pi_DTW)
        C1 = torch.matmul(out_conv_source_sq[idx_cl], torch.sum(pi_DTW, dim=1)).sum(-1)
        C2 = torch.matmul(out_conv_target_sq, torch.sum(pi_DTW.T, dim=1)).sum(-1)
        C3 = torch.tensordot(torch.matmul(out_conv_source[idx_cl], pi_DTW), out_conv_target, dims=([1, 2], [1, 2]))
        C4 = C1[:, None] + C2[None, :] - 2 * C3
        global_l2_matrix = C4

        for cl in range(1, loop_iteration):
            idx_cl = torch.where(labels_source == cl)
            pi_DTW = self.DTW[cl]
            pi_DTW = torch.tensor(pi_DTW)
            C1 = torch.matmul(out_conv_source_sq[idx_cl], torch.sum(pi_DTW, dim=1)).sum(-1)
            C2 = torch.matmul(out_conv_target_sq, torch.sum(pi_DTW.T, dim=1)).sum(-1)
            C3 = torch.tensordot(torch.matmul(out_conv_source[idx_cl], pi_DTW), out_conv_target, dims=([1, 2], [1, 2]))
            C4 = C1[:, None] + C2[None, :] - 2 * C3
            global_l2_matrix = torch.cat((global_l2_matrix, C4))
        l2_OT_matrix = OT * global_l2_matrix
        return l2_OT_matrix"""

    """def loss_CNN_MAD(self, labels_source, out_conv_source, out_conv_target):
        
        Compute the loss for the CNN part of CNN-MAD
        :param labels: the true labels of the source batch
        :param OT: An OT matrix from MAD
        :param DTW: DTW matrices from MAD, one per class
        :param out_source: out of the convolutions for the source batch
        :param out_target: out of the convolutions for the target batch
        :param softmax_target: out of the classifier for the target batch
        :return: The loss

        Y0 = torch.zeros(size=(5))
        Y1 = torch.zeros(size=(5))
        labels = torch.cat((Y0, Y1), 0)
        OT = torch.eyes(size=(10, 10))
        DTW = [torch.eyes(size=(20, 20))]
        out_source = torch.ones(size=(10, 20, 2))
        out_target = torch.zeros(size=(10, 20, 2))
        t0 = torch.zeros(size=(10))
        t1 = torch.ones(size(10))
        softmax_target = torch.cat((t0, t1), 1)

        Loss = loss_CNN_MAD(labels, OT, DTW, out_source, out_target, softmax_target)
        examples
        --------
        >>> labels0 = torch.zeros(size=(100,)).type(torch.long)
        >>> labels1 = torch.ones(size=(100,)).type(torch.long)
        >>> series0_source0 = torch.zeros(size=(100, 20, 1)).type(torch.float)
        >>> series0_source1 = torch.ones(size=(100, 80, 1)).type(torch.float)
        >>> series0_source = torch.cat((series0_source0, series0_source1), dim=1)
        >>> series1_source0 = torch.zeros(size=(100, 80, 1)).type(torch.float)
        >>> series1_source1 = torch.ones(size=(100, 20, 1)).type(torch.float)
        >>> series1_source = torch.cat((series1_source0, series1_source1), dim=1)
        >>> series_source = torch.cat((series0_source, series1_source), dim=0)
        >>> series0_target0 = torch.zeros(size=(100, 40, 1)).type(torch.float)
        >>> series0_target1 = torch.ones(size=(100, 60, 1)).type(torch.float)
        >>> series0_target = torch.cat((series0_target0, series0_target1), dim=1)
        >>> series1_target0 = torch.zeros(size=(100, 60, 1)).type(torch.float)
        >>> series1_target1 = torch.ones(size=(100, 40, 1)).type(torch.float)
        >>> series1_target = torch.cat((series1_target0, series1_target1), dim=1)
        >>> series_target = torch.cat((series0_target, series1_target), dim=0)
        >>> series_target.requires_grad = True
        >>> labels = torch.cat((labels0, labels1))
        >>> cnnmad = CNNMAD(name="", batchsize=10, beta=1.0, alpha=1.0, channel=1, valid_target_label=labels, train_target_data=series_target, test_target_data=series_target, test_target_label=labels, test_source_label=labels, test_source_data=series_source, train_source_data=series_source, train_source_label=labels, valid_source_data=series_source, valid_target_data=series_target , num_class=2, valid_source_label=labels)
        >>> cnnmad.set_current_batchsize(200)
        >>> cnnmad.set_current_batchsize(200, train=False)
        >>> out_source, out_conv_source = cnnmad.forward(cnnmad.trainSourceData.transpose(1, 2))
        >>> out_target, out_conv_target = cnnmad.forward(cnnmad.trainTargetData.transpose(1, 2), train=False)
        >>> cnnmad.mad(out_conv_source=out_conv_source, out_conv_target=out_conv_target, labels_source=labels)
        >>> out_cnn = cnnmad.loss_CNN_MAD(labels_source=labels, out_conv_source=out_conv_source, out_conv_target=out_conv_target, logSoftmax_target=out_target)
        >>> out_cnn.backward()
        >>> print(series_target.grad)
        tensor(0.8064, dtype=torch.float64, grad_fn=<SumBackward0>)

        OT = torch.tensor(self._OT)
        if self.CUDA_train:
            if torch.cuda.is_available():
                OT = OT.cuda()

        if self.MAD_class:
            loop_iteration = torch.max(labels_source).item() + 1
        else:
            loop_iteration = 1
        alpha_loss = self.l2_torch(labels_source=labels_source, out_conv_source=out_conv_source,
                                   loop_iteration=loop_iteration, OT=OT, out_conv_target=out_conv_target)

        self.loss_alpha.append(alpha_loss.sum().item())
        return alpha_loss.sum()"""

    def g(self, x):
        """

        :param x:
        :return:
        examples
        --------
        >>> labels0 = torch.zeros(size=(100,)).type(torch.long)
        >>> labels1 = torch.ones(size=(100,)).type(torch.long)
        >>> series0_source0 = torch.zeros(size=(100, 20, 1)).type(torch.float)
        >>> series0_source1 = torch.ones(size=(100, 80, 1)).type(torch.float)
        >>> series0_source = torch.cat((series0_source0, series0_source1), dim=1)
        >>> series1_source0 = torch.zeros(size=(100, 80, 1)).type(torch.float)
        >>> series1_source1 = torch.ones(size=(100, 20, 1)).type(torch.float)
        >>> series1_source = torch.cat((series1_source0, series1_source1), dim=1)
        >>> series_source = torch.cat((series0_source, series1_source), dim=0)
        >>> labels = torch.cat((labels0, labels1))
        >>> cnnmad = CNNMAD(name="", batchsize=10, beta=0.0, alpha=0.0, channel=1, train_source_data=series_source, train_source_label=labels, num_class=2)
        >>> out = cnnmad.g(series_source.transpose(2, 1))
        >>> print(out.shape)
        torch.Size([200, 128, 100])
        """
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        return out

    def forward(self, x):
        out_conv = self.g(x)
        return out_conv

    def train_iteration(self):
        """
        Main function of the model
        Trains an iteration for both regular CNN and MADCNN (thanks to some if/else)

        examples
        --------
        >>> labels0 = torch.zeros(size=(100,)).type(torch.long)
        >>> labels1 = torch.ones(size=(100,)).type(torch.long)
        >>> series0_source0 = torch.rand(size=(100, 20, 1)).type(torch.float)
        >>> series0_source1 = torch.rand(size=(100, 80, 1)).type(torch.float)
        >>> series0_source = torch.cat((series0_source0, series0_source1), dim=1)
        >>> series1_source0 = torch.rand(size=(100, 80, 1)).type(torch.float)
        >>> series1_source1 = torch.rand(size=(100, 20, 1)).type(torch.float)
        >>> series1_source = torch.cat((series1_source0, series1_source1), dim=1)
        >>> series_source = torch.cat((series0_source, series1_source), dim=0)
        >>> series0_target0 = torch.rand(size=(100, 40, 1)).type(torch.float)
        >>> series0_target1 = torch.rand(size=(100, 60, 1)).type(torch.float)
        >>> series0_target = torch.cat((series0_target0, series0_target1), dim=1)
        >>> series1_target0 = torch.rand(size=(100, 60, 1)).type(torch.float)
        >>> series1_target1 = torch.rand(size=(100, 40, 1)).type(torch.float)
        >>> series1_target = torch.cat((series1_target0, series1_target1), dim=1)
        >>> series_target = torch.cat((series0_target, series1_target), dim=0)
        >>> labels = torch.cat((labels0, labels1))
        >>> series_source.requires_grad = True
        >>> series_target.requires_grad = True
        >>> cnnmad = CNNMAD(name="", batchsize=2, beta=0.0, alpha=1.0, lamb=0.0, channel=1, valid_target_data=series_target, valid_target_label=labels, valid_source_data=series_source, valid_source_label=labels, test_target_label=labels, test_source_data=series_source, test_target_data=series_target, train_target_data=series_target, train_source_data=series_source, train_source_label=labels, num_class=2)
        >>> before_classif = list(cnnmad.classifier.parameters()).copy()[0][0]
        >>> before_conv = list(cnnmad.conv1.parameters()).copy()[0][0]
        >>> print(before_conv)
        >>> cnnmad.fit_madcnn(iterations=2, validation=100)
        >>> after_classif = list(cnnmad.classifier.parameters())[0][0]
        >>> after_conv = list(cnnmad.conv1.parameters())[0][0]
        >>> print(after_conv)
        tensor(True)
        """
        self.train()
        print(self.iteration)
        self.new_iteration()
        index = self.mini_batch_class_balanced()
        self.optimizer.zero_grad()
        inputs_source = torch.tensor(self.trainSourceData[index]).type(torch.float)
        labels_source = torch.tensor(self.trainSourceLabel[index])
        if self.CUDA_train:
            if torch.cuda.is_available():
                inputs_source = inputs_source.cuda()
                labels_source = labels_source.cuda()
        self.set_current_batchsize(inputs_source.shape[0])
        out_conv_source = self.forward(inputs_source.transpose(1, 2))
        random_index = np.random.choice(self.trainTargetData.shape[0], self.batchsize)
        inputs_target = torch.tensor(self.trainTargetData[random_index]).type(torch.float)
        if self.CUDA_train:
            if torch.cuda.is_available():
                inputs_target = inputs_target.cuda()
        self.set_current_batchsize(inputs_target.shape[0], train=False)
        out_conv_target = self.forward(inputs_target.transpose(1, 2))
        """self.mad(out_conv_source, out_conv_target, labels_source)
        loss_cnn_mad = self.loss_CNN_MAD(labels_source, out_conv_source, out_conv_target)"""
        loss_cnn_mad, self._OT, self._DTW, self._Cost_OT = self.MAD_loss_train.loss_CNN_MAD(out_conv_source,
                                                                                            out_conv_target,
                                                                                            labels_source)
        l2_lambda = self.reg
        l2_reg = torch.tensor(0.)
        if self.reg_rep:
            l2_reg += torch.norm(out_conv_source) ** 2
        else:
            l2_reg += torch.norm(self.conv1[0].weight) ** 2
        loss = loss_cnn_mad - l2_lambda * l2_reg
        self.norm_conv1.append(l2_reg.detach().item())
        self.loss_count.append(loss.detach().item())
        self.loss_alpha.append(loss_cnn_mad.item())
        self.weight_con1.append(self.conv1[0].weight.detach().numpy().copy())
        loss.backward()
        self.optimizer.step()
        print(self.conv1[0].weight[0:10])

    def fit_cnn(self, cnn_iterations, validation):
        """
        Train a regular CNN
        :parameter
        cnn_iterations : number of training iterations
        validation : every how many steps for validation
        examples
        --------
        >>> import numpy.random as npr
        >>> npr.seed(1)
        >>> labels0 = torch.zeros(size=(100,)).type(torch.long)
        >>> labels1 = torch.ones(size=(100,)).type(torch.long)
        >>> series0 = torch.zeros(size=(100, 100, 1), dtype=float)
        >>> series1 = torch.ones(size=(100, 100, 1), dtype=float)
        >>> series = torch.cat((series0, series1))
        >>> labels = torch.cat((labels0, labels1))
        >>> cnnmad = CNNMAD(name="", batchsize=10, channel=1, test_target_data=series, test_target_label=labels, test_source_data=series, train_source_data=series, valid_source_data=series, valid_target_data=series, train_source_label=labels, num_class=3, valid_source_label=labels)
        >>> cnnmad.fit(iterations=0, cnn_iterations=10)
        CNN only training final evaluation on test target dataset best model at iteration  0
        Average loss: 0.0033, Accuracy: 200/200 (100.000%)
        MAD-CNN training final evaluation on test target dataset best model at iteration  0
        Average loss: 0.0033, Accuracy: 200/200 (100.000%)
        0
        """
        while self.iteration < cnn_iterations:
            self.train_iteration()
            current_loss, _ = self.valid_epoch(domain="source", validation=validation)
            self.loss_count_cnn_valid_source.append(current_loss)
            if len(self.loss_count_cnn_valid_source) > 21:
                mean_loss = torch.mean(torch.tensor(self.loss_count_cnn_valid_source[-20:]))
                former_mean_loss = torch.mean(torch.tensor(self.loss_count_cnn_valid_source[-21:-1]))
                if mean_loss < former_mean_loss:
                    self.best_iteration_cnn = self.iteration
                    self.best_model_cnn = self.state_dict()
                elif self.iteration - self.best_iteration_cnn > 50:
                    self.iteration = cnn_iterations
                    self.save_best_model(cnn=True)
            if self.iteration % validation == 0:
                self.save_best_model(cnn=True)
                self.evaluate()

    def fit_madcnn(self, iterations, validation=10000):
        """
        Train the MAD-CNN model
        :parameter
        iterations : number of iterations during tranning
        validation : how many steps between validation
        examples
        --------
        >>> labels0 = torch.zeros(size=(100,)).type(torch.long)
        >>> labels1 = torch.ones(size=(100,)).type(torch.long)
        >>> series0_source0 = torch.zeros(size=(100, 20, 1), dtype=float)
        >>> series0_source1 = torch.ones(size=(100, 80, 1), dtype=float)
        >>> series0_source = torch.cat((series0_source0, series0_source1), dim=1)
        >>> series1_source0 = torch.zeros(size=(100, 80, 1), dtype=float)
        >>> series1_source1 = torch.ones(size=(100, 20, 1), dtype=float)
        >>> series1_source = torch.cat((series1_source0, series1_source1), dim=1)
        >>> series_source = torch.cat((series0_source, series1_source), dim=0)
        >>> series0_target0 = torch.zeros(size=(100, 40, 1), dtype=float)
        >>> series0_target1 = torch.ones(size=(100, 60, 1), dtype=float)
        >>> series0_target = torch.cat((series0_target0, series0_target1), dim=1)
        >>> series1_target0 = torch.zeros(size=(100, 60, 1), dtype=float)
        >>> series1_target1 = torch.ones(size=(100, 40, 1), dtype=float)
        >>> series1_target = torch.cat((series1_target0, series1_target1), dim=1)
        >>> series_target = torch.cat((series0_target, series1_target), dim=0)
        >>> labels = torch.cat((labels0, labels1))
        >>> cnnmad = CNNMAD(name="", batchsize=10, channel=1, train_target_data=series_target, train_source_data=series_source, train_source_label=labels, num_class=2)
        >>> cnnmad.fit(iterations=10, cnn_iterations=0)
        CNN only training final evaluation on test target dataset best model at iteration  0
        Average loss: 0.0035, Accuracy: 100/200 (50.000%)
        MAD-CNN training final evaluation on test target dataset best model at iteration  0
        Average loss: 0.0024, Accuracy: 200/200 (100.000%)
        10
        """
        while self.iteration < iterations:
            self.train_iteration()
            """if len(self.loss_count_valid_target) > 21:
                mean_loss = np.mean(self.loss_count_valid_target[-20:])
                former_mean_loss = np.mean(self.loss_count_valid_target[-21:-1])
                if mean_loss < former_mean_loss:
                    self.best_iteration = self.iteration
                    self.best_model = self.state_dict()
                elif self.iteration - self.best_iteration > 2000:
                    self.iteration = iterations
                    self.save_best_model()"""
            if self.iteration == validation:
                names = ["loss_alpha.npy", "norm_conv1.npy", "weight_conv1.npy"]
                files = [self.loss_alpha, self.norm_conv1, self.weight_con1]
                self.save_stuff(names=names, files=files)
                self.forward_MAD()
                self.forward_MAD(dataset="train")
        if self.saving:
            names = ["loss_alpha.npy", "norm_conv1.npy", "weight_conv1.npy"]
            files = [self.loss_alpha, self.norm_conv1, self.weight_con1]
            self.save_stuff(names=names, files=files)
        self.forward_MAD()
        self.forward_MAD(dataset="train")

    def fit(self, iterations, cnn_iterations=0, validation=10000):
        """
        :parameter
        iterations = number of iterations on MAD CNN training
        cnn_iterations = number of iterations on CNN with no domain adaptation
        validation = every how many iterations would you like to validate the model
        :return
        Nothing

        examples
        --------
        Necessary ?
        Probably not
        """

        self.fit_cnn(cnn_iterations=cnn_iterations, validation=validation)

        self.iteration = 0
        self.fit_madcnn(iterations=iterations)

    def new_iteration(self):
        self.iteration += 1

    def valid_epoch(self, domain, validation=1):
        """
        Validate the model on the validation set either of the source or target domain
        :param domain: which domain to choose
        :return: nothing but print enlightning resutls
        examples
        ---------
        >>> labels0 = torch.zeros(size=(100,)).type(torch.long)
        >>> labels1 = torch.ones(size=(100,)).type(torch.long)
        >>> series0_source0 = torch.zeros(size=(100, 20, 1), dtype=float)
        >>> series0_source1 = torch.ones(size=(100, 80, 1), dtype=float)
        >>> series0_source = torch.cat((series0_source0, series0_source1), dim=1)
        >>> series1_source0 = torch.zeros(size=(100, 80, 1), dtype=float)
        >>> series1_source1 = torch.ones(size=(100, 20, 1), dtype=float)
        >>> series1_source = torch.cat((series1_source0, series1_source1), dim=1)
        >>> series_source = torch.cat((series0_source, series1_source), dim=0)
        >>> series0_target0 = torch.zeros(size=(100, 40, 1), dtype=float)
        >>> series0_target1 = torch.ones(size=(100, 60, 1), dtype=float)
        >>> series0_target = torch.cat((series0_target0, series0_target1), dim=1)
        >>> series1_target0 = torch.zeros(size=(100, 60, 1), dtype=float)
        >>> series1_target1 = torch.ones(size=(100, 40, 1), dtype=float)
        >>> series1_target = torch.cat((series1_target0, series1_target1), dim=1)
        >>> series_target = torch.cat((series0_target, series1_target), dim=0)
        >>> labels = torch.cat((labels0, labels1))
        >>> labels_target = torch.cat((labels0, labels0))
        >>> cnnmad = CNNMAD(name="", batchsize=10, beta=0.0, alpha=0.0, channel=1, valid_target_label=labels_target, train_target_data=series_target, test_target_data=series_target, test_target_label=labels_target, test_source_label=labels, test_source_data=series_source, train_source_data=series_source, train_source_label=labels, valid_source_data=series_source, valid_target_data=series_target , num_class=2, valid_source_label=labels)
        >>> cnnmad.valid_epoch(domain="target")
        target :
        Average loss: 0.0031, Accuracy: 200/200 (100.000%)
        (tensor(0.0031), 200)
        """
        self.eval()
        with torch.no_grad():
            if domain == "target":
                inputs = torch.tensor(self.validTargetData).type(torch.float)
                labels = torch.tensor(self.validTargetLabel)
                if self.CUDA_train:
                    if torch.cuda.is_available():
                        inputs = inputs.cuda()
                        labels = labels.cuda()

            if domain == "source":
                inputs = torch.tensor(self.validSourceData).type(torch.float)
                labels = torch.tensor(self.validSourceLabel)
                if self.CUDA_train:
                    if torch.cuda.is_available():
                        inputs = inputs.cuda()
                        labels = labels.cuda()
            self.set_current_batchsize(inputs.shape[0])
            out, out_cnn = self.forward(inputs.transpose(1, 2))
            loss = self.crossLoss(out.float(), labels)
            pred = out.data.max(1, keepdim=True)[1]

            correct = pred.eq(labels.data.view_as(pred)).cpu().sum()
            len_data = len(labels)
            loss /= len_data
            self.accuracy = 100. * correct / len_data
            if domain == 'source':
                self.acc_source.append(100. * correct / len_data)
            else:
                self.acc_target.append(100. * correct / len_data)
            if self.iteration % validation == 0:
                print(self.iteration, "Validation set ", domain, ":")
                print('Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)'.format(loss, correct, len_data,
                                                                               100. * correct / len_data))
            return loss, len_data

    def evaluate(self, domain="target"):
        """
        Evaluate the method on the test dataset either of the source or target domain
        :param:
        domain : in which domain the model is evaluated
        :return: Nothing but does some printing along with saving various stuff
        examples
        --------
        >>> labels0 = torch.zeros(size=(100,)).type(torch.long)
        >>> labels1 = torch.ones(size=(100,)).type(torch.long)
        >>> series0_source0 = torch.zeros(size=(100, 20, 1), dtype=float)
        >>> series0_source1 = torch.ones(size=(100, 80, 1), dtype=float)
        >>> series0_source = torch.cat((series0_source0, series0_source1), dim=1)
        >>> series1_source0 = torch.zeros(size=(100, 80, 1), dtype=float)
        >>> series1_source1 = torch.ones(size=(100, 20, 1), dtype=float)
        >>> series1_source = torch.cat((series1_source0, series1_source1), dim=1)
        >>> series_source = torch.cat((series0_source, series1_source), dim=0)
        >>> series0_target0 = torch.zeros(size=(100, 40, 1), dtype=float)
        >>> series0_target1 = torch.ones(size=(100, 60, 1), dtype=float)
        >>> series0_target = torch.cat((series0_target0, series0_target1), dim=1)
        >>> series1_target0 = torch.zeros(size=(100, 60, 1), dtype=float)
        >>> series1_target1 = torch.ones(size=(100, 40, 1), dtype=float)
        >>> series1_target = torch.cat((series1_target0, series1_target1), dim=1)
        >>> series_target = torch.cat((series0_target, series1_target), dim=0)
        >>> labels = torch.cat((labels0, labels1))
        >>> labels_target = torch.cat((labels0, labels0))
        >>> cnnmad = CNNMAD(name="", batchsize=10, beta=0.0, alpha=0.0, channel=1, valid_target_label=labels_target, train_target_data=series_target, test_target_data=series_target, test_target_label=labels_target, test_source_label=labels, test_source_data=series_source, train_source_data=series_source, train_source_label=labels, valid_source_data=series_source, valid_target_data=series_target , num_class=2, valid_source_label=labels)
        >>> cnnmad.evaluate()
        Average loss: 0.0031, Accuracy: 200/200 (100.000%)
        """
        with torch.no_grad():
            self.eval()
            if domain == "target":
                inputs = torch.tensor(self.testTargetData).type(torch.float)
                labels = torch.tensor(self.testTargetLabel)
                if self.CUDA_train:
                    if torch.cuda.is_available():
                        inputs = inputs.cuda()
                        labels = labels.cuda()
            if domain == "source":
                inputs = torch.tensor(self.testSourceData).type(torch.float)
                labels = torch.tensor(self.testSourceLabel)
                if self.CUDA_train:
                    if torch.cuda.is_available():
                        inputs = inputs.cuda()
                        labels = labels.cuda()
            with torch.no_grad():
                self.set_current_batchsize(inputs.shape[0])
                out, out_cnn = self.forward(inputs.transpose(1, 2))
                out_cnn_mean = out_cnn.mean(2)
                loss = self.crossLoss(out.float(), labels)
                pred = out.data.max(1, keepdim=True)[1]
                correct = pred.eq(labels.data.view_as(pred)).cpu().sum()
                names = [domain + "_rout_conv.npy", domain + "_out_conv.npy", domain + "_prediction.npy",
                         domain + "_target.npy", domain + "_confusion_mat.npy"]
                files = [out_cnn.cpu(), out_cnn_mean.cpu(), pred.cpu(), labels.cpu(),
                         sklearn.metrics.confusion_matrix(labels.cpu(), pred.cpu())]
                self.save_stuff(files=files, names=names)
                loss /= len(labels)
            print(self.iteration, "Evaluation set ", domain, ":")
            print('Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)'.format(loss, correct, len(labels),
                                                                           100. * correct / len(labels)))

    def set_current_batchsize(self, dim, train=True):
        if train:
            self.current_batchtrain_ = dim
        else:
            self.current_batchtest_ = dim

    def set_name(self, new_name):
        self.name = new_name

    def forward_MAD(self, dataset="valid", SourceTarget=True):
        """
        Make a forward pass on any dataset and for source-source or source-target domains in order to get and save the
        ouputs of the network and the DTW, OT and OT cost from MAD
        :param dataset: on which dataset to perform
        :param SourceTarget: if true perform the forward for source->target domains else does source->source
        :return: the DTW, OT and OT cost from MAD and save the output from the network
        examples
        --------
        >>> labels = torch.zeros(size=(4,)).type(torch.long)
        >>> series_source0 = torch.zeros(size=(4, 1, 1), dtype=float)
        >>> series_source1 = torch.ones(size=(4, 3, 1), dtype=float)
        >>> series_source = torch.cat((series_source0, series_source1), dim=1)
        >>> series_target0 = torch.zeros(size=(4, 3, 1), dtype=float)
        >>> series_target1 = torch.ones(size=(4, 1, 1), dtype=float)
        >>> series_target = torch.cat((series_target0, series_target1), dim=1)
        >>> cnnmad = CNNMAD(name="", batchsize=10, beta=1.0, alpha=1.0, channel=1, valid_target_label=labels, train_target_data=series_target, test_target_data=series_target, test_target_label=labels, test_source_label=labels, test_source_data=series_source, train_source_data=series_source, train_source_label=labels, valid_source_data=series_source, valid_target_data=series_target , num_class=1, valid_source_label=labels)
        >>> cnnmad.forward_MAD()
        ([array([[1., 0., 0., 0.],
               [0., 1., 0., 0.],
               [0., 0., 1., 0.],
               [0., 0., 0., 1.]])], array([[0.25, 0.  , 0.  , 0.  ],
               [0.  , 0.  , 0.25, 0.  ],
               [0.  , 0.25, 0.  , 0.  ],
               [0.  , 0.  , 0.  , 0.25]]), array([[39.5, 39.5, 39.5, 39.5],
               [39.5, 39.5, 39.5, 39.5],
               [39.5, 39.5, 39.5, 39.5],
               [39.5, 39.5, 39.5, 39.5]]))
        """
        self.eval()
        with torch.no_grad():
            if dataset == "train":
                inputs_source = self.trainSourceData.float()
                labels_source = self.trainSourceLabel
                inputs_target = self.trainTargetData.float()
                labels_target = self.trainTargetLabel
                train_test = "train"
            if dataset == 'test':
                inputs_source = self.testSourceData.float()
                labels_source = self.testSourceLabel
                inputs_target = self.testTargetData.float()
                labels_target = self.testTargetLabel
                train_test = "test"
            if dataset == "valid":
                inputs_source = self.validSourceData.float()
                labels_source = self.validSourceLabel
                inputs_target = self.validTargetData.float()
                labels_target = self.validTargetLabel
                train_test = "valid"

            out_conv_source = self.forward(inputs_source.transpose(1, 2))
            if SourceTarget:
                out_conv_target = self.forward(inputs_target.transpose(1, 2))
            else:
                out_conv_target = out_conv_source
            loss_mad_valid, self._OT_valid, self.DTW_valid, self._Cost_OT_valid = \
                self.MAD_loss_valid.loss_CNN_MAD(out_conv_source, out_conv_target, labels_source)
            names = [train_test + 'DTW_forward_MAD.npy', train_test + 'OT_forward_MAD.npy',
                     train_test + 'OT_Cost_forward_MAD.npy', train_test + 'Conv_target.npy',
                     train_test + "Conv_source.npy", train_test + 'labels_target.npy', train_test + 'labels_source.npy']
            files = [self.DTW_valid, self._OT_valid, self._cost_OT_valid, out_conv_target.detach().numpy(),
                     out_conv_source.detach().numpy(), labels_target, labels_source]
            self.save_stuff(names=names, files=files)

    def save_stuff(self, files, names):
        for stuff in range(0, len(files)):
            np.save(self.name + str(self.iteration) + names[stuff], files[stuff])

    def save_best_model(self, cnn=False):
        if cnn:
            torch.save(self.best_model_cnn, self.name + str(self.best_iteration) + 'cnn.pt')
        else:
            torch.save(self.best_model, self.name + str(self.best_iteration) + '.pt')

