from numpy.core.fromnumeric import mean
import sklearn.metrics
import torch
import time
import torch.nn as nn
import numpy.random as npr
import argparse
from torch.utils.data import Dataset
import numpy as np
import threading
import warnings
from MAD_loss import MAD_loss


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
    def __init__(self, name, batchsize, n_classes,
                 feature_extractor, classifier,
                 alpha=0.01, beta=0.01, lamb=1.,
                 MAD_class=True, supervised_validation=False,
                 lr=0.001, saving=False, 
                 CUDA_train=False,
                 CUDA_MAD=False,
                 dropout=False,
                 torching=True,
                 target_prop=None,
                 batch_or_mass=True,
                 unbalanced=False,
                 un_reg=None,
                 un_mas=1.0
                 ):
        super(CNNMAD, self).__init__()
        self.n_classes = n_classes
        self.CUDA_train = CUDA_train
        self.CUDA_MAD = CUDA_MAD
        self.name = name
        self.target_prop = target_prop
        self.lamb = lamb
        self.alpha = alpha
        self.beta = beta
        self.alpha_first = alpha
        self.beta_first = beta
        self.lr = lr
        self.saving = saving
        self.last_accuracy = 0
        self.DTW = None
        self.batchsize = batchsize
        self.MAD_class = MAD_class
        self.unbalanced = unbalanced
        self.gen = torch.Generator()
        self.feature_extractor = feature_extractor
        self.classifier = classifier
        self.logSoftmax = nn.LogSoftmax(dim=1)
        self.dp = dropout
        if dropout:
            self.dropout = nn.Dropout(0.25)
        self.supervised_validation = supervised_validation
        torch.nn.init.xavier_uniform_(self.feature_extractor[0].weight)
        torch.nn.init.xavier_uniform_(self.feature_extractor[3].weight)
        torch.nn.init.xavier_uniform_(self.feature_extractor[6].weight)
        torch.nn.init.xavier_uniform_(self.classifier[0].weight)
        if (self.alpha == 0) and (self.beta != 0):
            self.optimizer = torch.optim.Adam([{'params': self.classifier.parameters()}],
                                              lr=lr, amsgrad=True)
        else:
            self.optimizer = torch.optim.Adam([{'params': self.feature_extractor.parameters()},
                                               {'params': self.classifier.parameters()}],
                                              lr=lr, amsgrad=True)

        self.crossLoss = nn.CrossEntropyLoss()
        self.batch_or_mass = batch_or_mass
        self.un_reg = un_reg
        self.un_mas = un_mas
        new_mass = None
        if self.batch_or_mass is False:
            new_mass = self.target_prop
        self.mad_loss = MAD_loss(n_classes, CUDA_train=self.CUDA_train, MAD_class=MAD_class, alpha=self.alpha,
                                 beta=self.beta, torching=torching, CUDA_MAD=self.CUDA_MAD,
                                 target_prop=new_mass, unbalanced=self.unbalanced, un_reg=self.un_reg, un_mas=self.un_mas)
        self.mad_loss_valid = MAD_loss(n_classes, CUDA_train=self.CUDA_train, MAD_class=MAD_class, alpha=self.alpha,
                                       beta=self.beta, torching=torching, CUDA_MAD=self.CUDA_MAD,
                                       target_prop=new_mass, unbalanced=self.unbalanced, un_reg=self.un_reg, un_mas=self.un_mas)
        self.iteration = 0
        self.best_iteration = 0
        self.loss_count = []
        self.loss_count_valid = []
        self.loss_beta = []
        self.loss_alpha = []
        self.acc_source = []
        self.acc_target = []
        if self.CUDA_train:
            print("Asked for Cuda")
            if torch.cuda.is_available():
                print('Cuda available')
                self.feature_extractor = self.feature_extractor.cuda()
                self.classifier = self.classifier.cuda()
                self.crossLoss = self.crossLoss.cuda()
                self.logSoftmax = self.logSoftmax.cuda()

    def mini_batch_class_balanced(self, X, vector, y=None, shuffle=True):
        """
        Draw minibatches with the same number of series for each classes
        :parameter
        shuffle : if true the dataset is shuffled before drawing
        :return
        the index of the series to pick
        examples
        --------
        >>> import numpy.random as npr
        >>> labels = npr.random_integers(low=0, high=2, size=100)
        >>> series = npr.random(size=(100, 100, 1))
        >>> cnnmad = CNNMAD(name="", batchsize=10, channel=1, train_source_data=series, train_source_label=labels, num_class=3)
        >>> index = cnnmad.mini_batch_class_balanced(shuffle=False)
        [0 0 0 1 1 1 2 2 2]
        """
        if y is not None:
            if shuffle:
                rindex = torch.randperm(len(X))
                X = X[rindex]
                y = y[rindex]
            index = torch.tensor([])
            if self.CUDA_train:
                if torch.cuda.is_available():
                    index = index.cuda()
            #  for i in range(self.n_classes):
            for i in torch.unique(y):
                s_index = torch.nonzero(y == i).squeeze()
                index_random = torch.randperm(n=s_index.shape[0], generator=self.gen)
                s_ind = s_index[index_random]
                index = torch.cat((index, s_ind[0:vector[i].item()]), 0)
            index = index.type(torch.long)
            index = index.view(-1)
            index_rand = torch.randperm(len(index), generator=self.gen)
            index = index[index_rand]
            #  sorting, index = torch.sort(index)
            X_minibatch, y_minibatch = X[index], y[index]
            """if self.CUDA_train:
                if torch.cuda.is_available():
                    X_minibatch = X_minibatch.cuda()
                    y_minibatch = y_minibatch.cuda()"""
            return X_minibatch.float(), y_minibatch.long(), index
        else:
            index = torch.randperm(len(X), generator=self.gen)
            index = index[:self.batchsize]
            X_target_minibatch = torch.tensor(X[index])
            """if self.CUDA_train:
                if torch.cuda.is_available():
                    X_target_minibatch = X_target_minibatch.cuda()"""
            return X_target_minibatch.float(), y, index

    def CE_similarity(self, labels_source, logSoftmax_target):
        def to_onehot(y, n_classe=0):
            ncl = torch.max(torch.tensor([torch.max(y), n_classe-1]))
            n_values = ncl + 1
            return torch.eye(n_values)[y]
        logSoftmax_target = self.logSoftmax(logSoftmax_target)
        labels_source_onehot = to_onehot(labels_source, n_classe=self.n_classes)
        if self.CUDA_train:
            if torch.cuda.is_available():
                labels_source_onehot = labels_source_onehot.cuda()
        logSoftmax_target_trans = torch.transpose(logSoftmax_target, 1, 0)
        similarity_cross_entropy = -torch.matmul(labels_source_onehot, logSoftmax_target_trans)
        return similarity_cross_entropy

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
        torch.Size([200, 128, 100])
        """
        return self.feature_extractor(x)

    def f(self, features_conv):
        """

        :param out:
        :param train:
        :return:
        examples
        --------
        >>> labels0 = torch.zeros(size=(100,)).type(torch.long)
        >>> labels1 = torch.ones(size=(100,)).type(torch.long)
        >>> series_source = torch.rand(size=(200, 128, 100))
        >>> labels = torch.cat((labels0, labels1))
        >>> cnnmad = CNNMAD(name="", batchsize=10, beta=0.0, alpha=0.0, channel=1, train_source_data=series_source, train_source_label=labels, num_class=2)
        >>> out = cnnmad.f(series_source)
        torch.Size([200, 2])
        """
        h = features_conv.mean(dim=2, keepdim=False)  # Average pooling
        h = self.classifier(h)
        log_probas = self.logSoftmax(h)
        return log_probas

    def forward(self, x):
        features_conv = self.g(x)
        if self.dp:
            self.dropout(features_conv)
        log_probas = self.f(features_conv)
        return log_probas, features_conv

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
        >>> cnnmad.fit_madcnn(iterations=2, validation=100)
        >>> after_classif = list(cnnmad.classifier.parameters())[0][0]
        >>> after_conv = list(cnnmad.conv1.parameters())[0][0]
        tensor(True)
        """
        self.train()
        self.new_iteration()
        self.optimizer.zero_grad()
        X_source_minibatch, y_source_minibatch, index_source = self.mini_batch_class_balanced(X=self.X_source_,
                                                                                              vector=self.sample_vec_train,
                                                                                              y=self.y_source_,
                                                                                              shuffle=True)
        X_target_minibatch, _, index_target = self.mini_batch_class_balanced(self.X_target_,
                                                                             vector=self.sample_vec_train,
                                                                             shuffle=True)
        loss, _ = self.compute_total_loss(X_source_minibatch, y_source_minibatch, X_target_minibatch)
        if loss > 250000:
            self.save_best_model()
        loss.backward()
        #  torch.nn.utils.clip_grad_value_(self.feature_extractor.parameters(), clip_value=0.1)
        self.optimizer.step()
    
    def compute_total_loss(self, X_source, y_source, X_target, training=True):
        logprobas_source, conv_features_source = self.forward(X_source.transpose(1, 2))
        classif_loss = self.crossLoss(logprobas_source, y_source)
        if training:
            self.loss_count.append(classif_loss.item())
        loss = self.lamb * classif_loss
        if (self.alpha != 0) or (self.beta != 0):
            logprobas_target, conv_features_target = self.forward(X_target.transpose(1, 2))
            similarity_CE = self.CE_similarity(labels_source=y_source, logSoftmax_target=logprobas_target)
            alpha_loss, beta_loss = self.mad_loss(conv_features_source, conv_features_target, y_source, similarity_CE,
                                                  self.sample_vec_train)
            if training:
                self.loss_alpha.append(alpha_loss.item())
                self.loss_beta.append(beta_loss.item())
            loss += alpha_loss + beta_loss
        """if self.beta != 0:
            logprobas_target, conv_features_target = self.forward(X_target.transpose(1, 2))
            similarity_CE = self.CE_similarity(labels_source=y_source, logSoftmax_target=logprobas_target)
            beta_loss = self.mad_loss.forward_beta(similarity_CE, index_source, index_target)
            if training:
                self.loss_beta.append(beta_loss.item())
            loss += beta_loss"""
        return loss, logprobas_source

    def fit_several_epochs(self, max_iter, validation_step):
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
        former_mean_loss = np.inf
        n_iter_avg = 1000
        while self.iteration < max_iter:
            self.train_iteration()
            if self.iteration % 1 == 0:
                if self.supervised_validation:
                    valid_loss = self.supervised_validation_step(verbose_step=validation_step)
                else:
                    valid_loss = self.unsupervised_validation_step(verbose_step=validation_step)
                self.loss_count_valid.append(valid_loss.item())
            if self.iteration % n_iter_avg == 0 and self.iteration > 0:
                mean_loss = np.mean(self.loss_count_valid[-n_iter_avg:])
                if mean_loss < former_mean_loss:
                    former_mean_loss = mean_loss
                    self.best_iteration = self.iteration
                self.save_best_model()
                """elif self.iteration - self.best_iteration > 200:
                    break"""
            if self.iteration % validation_step == 0:
                print(self.iteration)
                if self.saving:
                    names = ["loss_train.npy", "loss_valid.npy", "loss_alpha.npy",
                             "loss_beta.npy", "acc_source.npy", "acc_target.npy"]
                    files = [self.loss_count, self.loss_count_valid,
                             self.loss_alpha, self.loss_beta, self.acc_source, self.acc_target]
                    self.save_stuff(names=names, files=files)
        if self.saving:
            names = ["loss_train.npy", "loss_valid.npy", "loss_alpha.npy",
                     "loss_beta.npy", "acc_source.npy", "acc_target.npy"]
            files = [self.loss_count, self.loss_count_valid, self.loss_alpha,
                     self.loss_beta, self.acc_source, self.acc_target]
            self.save_stuff(names=names, files=files)

    def fit(self, X_source, y_source, X_target, X_source_valid, y_source_valid, X_target_valid, y_target_valid,
            max_iter, n_iter_pretrain_cnn=0, validation_step=1000):
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
        self.X_source_ = X_source
        self.y_source_ = y_source
        self.X_target_ = X_target
        self.X_source_valid_ = X_source_valid
        self.y_source_valid_ = y_source_valid
        self.X_target_valid_ = X_target_valid
        self.y_target_valid_ = y_target_valid
        cl_u, count_classes = torch.unique(self.y_source_, return_counts=True)
        """sample_vec = torch.zeros(size=(self.n_classes,))
        while sample_vec.sum() < self.batchsize:
            for cl in range(0, self.n_classes):
                if (sample_vec[cl] < count_classes[cl]) and (sample_vec.sum() < self.batchsize):
                    sample_vec[cl] += 1
        self.sample_vec_train = sample_vec.type(torch.int)
        _, count_classes = torch.unique(self.y_source_valid_, return_counts=True)
        sample_vec = torch.zeros(size=(self.n_classes,))
        min_bs = torch.min(torch.tensor([self.batchsize, self.y_source_valid_.shape[0]]))
        while sample_vec.sum() < min_bs:
            for cl in range(0, self.n_classes):
                if sample_vec[cl] < count_classes[cl]:
                    sample_vec[cl] += 1
        self.sample_vec_valid = sample_vec.type(torch.int)"""
        sample_vec = torch.zeros(size=(self.n_classes,))
        #  for cl in range(0, self.n_classes):
        cl_ind = 0
        for cl in cl_u:
            if (self.target_prop is not None) and (self.batch_or_mass):
                cl_bs = torch.round(self.batchsize * self.target_prop[cl_ind])
            else:
                cl_bs = torch.round(self.batchsize * count_classes[cl_ind] / torch.sum(count_classes))
            if cl_bs <= 1:
                cl_bs = 2
            sample_vec[cl] = cl_bs
            cl_ind += 1
        while sample_vec.sum() > self.batchsize:
            sample_vec[torch.argmax(sample_vec)] -= 1
        while sample_vec.sum() < self.batchsize:
            sample_vec[torch.argmin(sample_vec)] += 1
        print(sample_vec, sample_vec.sum())
        self.sample_vec_train = sample_vec.type(torch.int)

        if self.y_source_valid_.shape[0] < self.batchsize:
            while sample_vec.sum() > self.y_source_valid_.shape[0]:
                sample_vec[torch.argmax(sample_vec)] -= 1
            while sample_vec.sum() < self.y_source_valid_.shape[0]:
                sample_vec[torch.argmin(sample_vec)] += 1

        self.sample_vec_valid = sample_vec.type(torch.int)

        if self.CUDA_train:
            if torch.cuda.is_available():
                self.X_source_ = self.X_source_.cuda()
                self.y_source_ = self.y_source_.cuda()
                self.X_target_ = self.X_target_.cuda()
                self.X_source_valid_ = self.X_source_valid_.cuda()
                self.y_source_valid_ = self.y_source_valid_.cuda()
                self.X_target_valid_ = self.X_target_valid_.cuda()
                self.y_target_valid_ = self.y_target_valid_.cuda()
        """if n_iter_pretrain_cnn > 0:
            self.set_loss_weight(alpha=0., beta=0.)
            self.fit_several_epochs(max_iter=n_iter_pretrain_cnn, validation_step=validation_step)"""

        if max_iter > 0:
            self.iteration = n_iter_pretrain_cnn
            self.set_loss_weight(alpha=self.alpha_first, beta=self.beta_first)
            self.fit_several_epochs(max_iter=max_iter, validation_step=validation_step)

    def new_iteration(self):
        self.iteration += 1

    def supervised_validation_step(self, verbose_step=1):
        """
        Validate the model on the validation set either of the source or target domain
        :param domain: which domain to choose
        :return: nothing but prnt enlightning resutls
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
        inputs, labels, index = self.mini_batch_class_balanced(X=self.X_target_valid_, vector=self.sample_vec_valid,
                                                               y=self.y_target_valid_, shuffle=False)
        log_probas, _ = self.forward(inputs.transpose(1, 2))
        loss = self.crossLoss(log_probas.float(), labels)
        pred = log_probas.data.max(1, keepdim=True)[1]
        correct = pred.eq(labels.data.view_as(pred)).cpu().sum()

        len_data = len(labels)
        self.accuracy = 100. * correct / len_data

        self.acc_target.append(100. * correct / len_data)
        if self.iteration % verbose_step == 0:
            print(self.iteration, "Validation set :")
            print('Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)'.format(loss, correct, len_data,
                                                                           100. * correct / len_data))
        return loss

    def unsupervised_validation_step(self, verbose_step=1):
        """
        Validate the model on the validation set either of the source or target domain
        :param domain: which domain to choose
        :return: nothing but prnt enlightning resutls
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
        inputs_source, labels_source, index_source = self.mini_batch_class_balanced(X=self.X_source_valid_,
                                                                                    vector=self.sample_vec_valid,
                                                                                    y=self.y_source_valid_,
                                                                                    shuffle=False)
        inputs_target, _, index_target = self.mini_batch_class_balanced(X=self.X_target_valid_,
                                                                        vector=self.sample_vec_valid, shuffle=False)
        if self.CUDA_train:
            if torch.cuda.is_available():
                inputs_target = inputs_target.cuda()
        loss, logprobas_source = self.compute_total_loss(inputs_source, labels_source, inputs_target, training=False)
        pred = logprobas_source.data.max(1, keepdim=True)[1]

        correct = pred.eq(labels_source.data.view_as(pred)).cpu().sum()
        len_data = len(labels_source)
        self.accuracy = 100. * correct / len_data
        self.acc_source.append(100. * correct / len_data)
        if self.iteration % verbose_step == 0:
            print(self.iteration, "Validation set :")
            print('Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)'.format(loss.detach().item(), correct, len_data,
                                                                           100. * correct / len_data))
        return loss

    def evaluate(self, inputs, labels, domain="target"):
        """
        Evaluate the method on the test dataset either of the source or target domain
        :param:
        domain : in which domain the model is evaluated
        :return: Nothing but does some prnting along with saving various stuff
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
            inputs = torch.tensor(inputs).type(torch.float)
            labels = torch.tensor(labels)
            if self.CUDA_train:
                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                    labels = labels.cuda()

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
        print(self.name)
        print(self.iteration, "Evaluation set ", domain, ":")
        print('Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)'.format(loss, correct, len(labels),
                                                                       100. * correct / len(labels)))
        print("F1 micro score is : ", sklearn.metrics.f1_score(labels.cpu(), pred.cpu(), average="micro"))
        print("F1 macro score is : ", sklearn.metrics.f1_score(labels.cpu(), pred.cpu(), average="macro"))
        print("F1 weigthed score is : ", sklearn.metrics.f1_score(labels.cpu(), pred.cpu(), average="weighted"))
        print(pred.shape, inputs.shape, labels.shape)
        return 100. * correct / len(labels), pred

    def proxy_labels(self, inputs):
        with torch.no_grad():
            self.eval()
            inputs = torch.tensor(inputs).type(torch.float)
            if self.CUDA_train:
                if torch.cuda.is_available():
                    inputs = inputs.cuda()
            out, out_cnn = self.forward(inputs.transpose(1, 2))
            pred = out.data.max(1, keepdim=True)[1]
        return pred

    def set_loss_weight(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def set_name(self, new_name):
        self.name = new_name

    def set_nb_classes(self, nb_classes, classifier=None):
        self.n_classes = nb_classes
        if classifier is not None:
            self.classifier = classifier
            torch.nn.init.xavier_uniform_(self.classifier[0].weight)

    def forward_MAD(self, X_source, y_source, X_target, y_target=None, train_test="valid", path=None):
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
        with torch.no_grad():
            self.eval()
            inputs_source = X_source.float()
            labels_source = y_source
            inputs_target = X_target.float()
            labels_target = y_target
            _, count_classes = torch.unique(labels_source, return_counts=True)
            sample_vec = torch.zeros(size=(self.n_classes,))
            for cl in range(0, self.n_classes):
                cl_bs = torch.round(self.batchsize * count_classes[cl] / torch.sum(count_classes))
                if cl_bs <= 1:
                    cl_bs += 1
                sample_vec[cl] = cl_bs
            while sample_vec.sum() > self.batchsize:
                sample_vec[torch.argmax(sample_vec)] -= 1
            sample_vec = sample_vec.type(torch.int)
            if self.CUDA_train:
                if torch.cuda.is_available():
                    inputs_source = inputs_source.cuda()
                    labels_source = labels_source.cuda()
                    inputs_target = inputs_target.cuda()
                    if labels_target is not None:
                        labels_target = labels_target.cuda()
            inputs_source, labels_source, index_source = self.mini_batch_class_balanced(X=inputs_source,
                                                                                        vector=sample_vec,
                                                                                        y=labels_source, shuffle=False)
            inputs_target, _, index_target = self.mini_batch_class_balanced(X=inputs_target,
                                                                            vector=sample_vec,
                                                                            shuffle=False)
            labels_target = labels_target[index_target]
            out_target, out_conv_target = self.forward(inputs_target.transpose(1, 2))
            out_source, out_conv_source = self.forward(inputs_source.transpose(1, 2))
            similarity_CE = self.CE_similarity(labels_source=labels_source, logSoftmax_target=out_target)
            self.mad_loss_valid.forward(out_conv_source, out_conv_target, labels_source, similarity_CE)
            pred = out_target.data.max(1, keepdim=True)[1]
            names = [train_test + 'DTW_forward_MAD.npy', train_test + 'OT_forward_MAD.npy',
                     train_test + 'OT_Cost_forward_MAD.npy', train_test + 'Conv_target.npy',
                     train_test + "Conv_source.npy", train_test + 'labels_target.npy', train_test + 'labels_source.npy',
                     train_test + "pred_forward.npy"]
            files = [self.torch2numpy(self.mad_loss_valid.DTW_), self.mad_loss_valid.OT_.cpu().numpy(),
                     self.mad_loss_valid.cost_OT_.cpu().numpy(),
                     out_conv_target.cpu().numpy(), out_conv_source.cpu().numpy(), labels_target.cpu().numpy(),
                     labels_source.cpu().numpy(),
                     pred.cpu().numpy()]
            self.save_stuff(names=names, files=files, path=path)

    def forward_OT(self, X_source, y_source, X_target):
        with torch.no_grad():
            self.eval()
            inputs_source = X_source.float()
            labels_source = y_source
            inputs_target = X_target.float()

            if self.CUDA_train:
                if torch.cuda.is_available():
                    inputs_source = inputs_source.cuda()
                    labels_source = labels_source.cuda()
                    inputs_target = inputs_target.cuda()
            _, count_classes = torch.unique(labels_source, return_counts=True)
            sample_vec = torch.zeros(size=(self.n_classes,))
            for cl in range(0, self.n_classes):
                cl_bs = torch.round(self.batchsize * count_classes[cl] / torch.sum(count_classes))
                if cl_bs <= 1:
                    cl_bs += 1
                sample_vec[cl] = cl_bs
            while sample_vec.sum() > self.batchsize:
                sample_vec[torch.argmax(sample_vec)] -= 1
            sample_vec = sample_vec.type(torch.int)
            inputs_source, labels_source, _ = self.mini_batch_class_balanced(X=inputs_source, y=labels_source, vector=sample_vec, shuffle=False)
            inputs_target, _, _ = self.mini_batch_class_balanced(X=inputs_target, y=None, vector=sample_vec, shuffle=False)
            out_source, out_conv_source = self.forward(inputs_source.transpose(1, 2))
            out_target, out_conv_target = self.forward(inputs_target.transpose(1, 2))
            similarity_CE = self.CE_similarity(labels_source=labels_source, logSoftmax_target=out_target)
            self.mad_loss.forward(out_conv_source, out_conv_target, labels_source, similarity_CE)
        return self.mad_loss.OT_

    def save_stuff(self, files, names, path=None):
        if path is None:
            path = self.name
        for stuff in range(0, len(files)):
            np.save(path + str(self.iteration) + names[stuff], files[stuff])

    def save_best_model(self):
        if self.best_iteration == self.iteration:
            torch.save(self.state_dict(), self.name + str(self.iteration) + 'best.pt')
        else:
            torch.save(self.state_dict(), self.name + str(self.iteration) + '.pt')

    @staticmethod
    def torch2numpy(list_):
        list_return = []
        for stuff in list_:
            list_return.append(stuff.cpu().numpy())
        return list_return


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-p1', '--pair1', type=int, help='The first half of the pair')
    parser.add_argument('-p2', '--pair2', type=int, help='The second half of the pair')
    parser.add_argument('-data', '--dataset', type=str, help="Which dataset to take between HAR and TarnBZH")
    parser.add_argument('-bs', '--batchsize', type=int, help='The batchsize')
    parser.add_argument("-a", "--alpha", type=float, help="Alpha")
    parser.add_argument("-b", "--beta", type=float, help='Beta')
    parser.add_argument('-lr', "--learning_rate", type=float, help="The learning rate")
    parser.add_argument('-e', "--epochs", type=int)
    parser.add_argument('-c', "--epochs_cnn", type=int)

    args, _ = parser.parse_known_args()


    def to_onehot(y):
        n_values = np.max(y) + 1
        return np.eye(n_values)[y]


    def from_numpy_to_torch(filename, float_or_long=True):
        data = np.load(filename)
        # data_t = torch.from_numpy(data)
        data_t = torch.tensor(data)
        if float_or_long:
            data_t = data_t.type(torch.float)
        else:
            data_t = data_t.type(torch.long)
        return data_t


    if args.dataset == 'HAR':
        chan = 9
        cn = 6

        source = args.pair1
        target = args.pair2

        # Train source dataset
        train_source = from_numpy_to_torch('/share/home/fpainblanc/MAD-CNN/numpy_data/ucihar_' + str(source) +
                                           'train.npy')
        train_source_label = from_numpy_to_torch('/share/home/fpainblanc/MAD-CNN/numpy_data/ucihar_' + str(source) +
                                                 'train_labels.npy', float_or_long=False)
        Data_source_train = Data_set(train_source, train_source_label)

        # Valid source dataset
        valid_source = from_numpy_to_torch('/share/home/fpainblanc/MAD-CNN/numpy_data/ucihar_' + str(source) +
                                           'valid.npy')
        valid_source_label = from_numpy_to_torch('/share/home/fpainblanc/MAD-CNN/numpy_data/ucihar_' + str(source) +
                                                 'valid_labels.npy', float_or_long=False)
        Data_source_valid = Data_set(valid_source, valid_source_label)

        # Test source dataset
        test_source = from_numpy_to_torch('/share/home/fpainblanc/MAD-CNN/numpy_data/ucihar_' + str(source) +
                                          'test.npy')
        test_source_label = from_numpy_to_torch('/share/home/fpainblanc/MAD-CNN/numpy_data/ucihar_' + str(source) +
                                                'test_labels.npy', float_or_long=False)
        Data_source_test = Data_set(test_source, test_source_label)

        # Train target dataset (no labels)
        train_target = from_numpy_to_torch('/share/home/fpainblanc/MAD-CNN/numpy_data/ucihar_' + str(target) +
                                           'train.npy')
        train_target_label = from_numpy_to_torch('/share/home/fpainblanc/MAD-CNN/numpy_data/ucihar_' + str(target) +
                                                 'train_labels.npy', float_or_long=False)
        Data_target_train = Data_set(train_target)

        # Valid target dataset
        valid_target = from_numpy_to_torch('/share/home/fpainblanc/MAD-CNN/numpy_data/ucihar_' + str(target) +
                                           'valid.npy')
        valid_target_label = from_numpy_to_torch('/share/home/fpainblanc/MAD-CNN/numpy_data/ucihar_' + str(target) +
                                                 'valid_labels.npy', float_or_long=False)
        Data_target_valid = Data_set(valid_target, valid_target_label)

        # Test target dataset
        test_target = from_numpy_to_torch('/share/home/fpainblanc/MAD-CNN/numpy_data/ucihar_' + str(target) +
                                          'test.npy')
        test_target_label = from_numpy_to_torch('/share/home/fpainblanc/MAD-CNN/numpy_data/ucihar_' + str(target) +
                                                'test_labels.npy', float_or_long=False)
        Data_target_test = Data_set(test_target, test_target_label)

    for batchsize in [256]:
        name1 = "hyperparameters/UCIHAR_CLASS/" + str(args.learning_rate) + "/" + str(batchsize) + "/" + \
               str(args.alpha) + "/" + str(args.beta) + "/" + str(args.pair1) + "_" + str(args.pair2) + "/HAR"
        name2 = "TarnBZH/" + str(args.learning_rate) + str(batchsize) + str(args.alpha) + str(args.beta) \
                + str(args.pair1) + "_" + str(args.pair2)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            """feature_extractor = nn.Sequential(
                nn.Conv1d(in_channels=chan, out_channels=128, kernel_size=8, stride=1, padding="same", bias=False),
                nn.BatchNorm1d(num_features=128, affine=False),
                nn.ReLU(),

                nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, stride=1, padding="same", bias=False),
                nn.BatchNorm1d(num_features=256, affine=False),
                nn.ReLU(),

                nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding="same", bias=False),
                nn.BatchNorm1d(num_features=128, affine=False),
                nn.ReLU()
            )

            classifier = nn.Sequential(nn.Linear(128, 6))  # /!\ Does not include softmax activation"""
            name = "hebo"
            CNN_mod = CNNMAD(name=name, batchsize=256,
                             # feature_extractor=feature_extractor, classifier=classifier,
                             n_classes=6, alpha=args.alpha, beta=args.beta, lamb=1.,
                             MAD_class=True, supervised_validation=True, lr=0.0001, saving=True, CUDA_train=True)

            CNN_mod.load_state_dict(torch.load("REG/basic_cnn/" + str(args.pair1) + "_" + str(args.pair2) +
                                               "/cnn_classif30000.pt"))
            CNN_mod.fit(X_source=train_source, y_source=train_source_label,
                        X_target=train_target, X_source_valid=valid_source,
                        y_source_valid=valid_source_label, X_target_valid=valid_target,
                        y_target_valid=valid_target_label, max_iter=args.epochs,
                        n_iter_pretrain_cnn=args.epochs_cnn, validation_step=1000)
            CNN_mod.evaluate(inputs=test_target, labels=test_target_label)
    # CNN_mod.evaluate(dataset_name='train')
