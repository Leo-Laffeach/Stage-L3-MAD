import torch
import torch.nn as nn
import numpy as np
import argparse
import os
import warnings
from MADCNN import CNNMAD as MADC
import sklearn.metrics


class CNNMAD(nn.Module):
    def __init__(self, name, batchsize, n_classes,
                 feature_extractor, classifier, supervised_validation=False,
                 lr=0.001, saving=False,
                 CUDA_train=False,
                 dropout=False,
                 target_prop=None
                 ):
        super(CNNMAD, self).__init__()
        self.n_classes = n_classes
        self.CUDA_train = CUDA_train
        self.name = name
        self.target_prop = target_prop
        self.lr = lr
        self.saving = saving
        self.last_accuracy = 0
        self.batchsize = batchsize
        self.gen = torch.Generator()
        self.feature_extractor = feature_extractor
        self.classifier = classifier
        print(nn, "#####################################")
        self.logSoftmax = nn.LogSoftmax(dim=1)
        self.dp = dropout
        if dropout:
            self.dropout = nn.Dropout(0.25)
        self.supervised_validation = supervised_validation
        torch.nn.init.xavier_uniform_(self.feature_extractor[0].weight)
        torch.nn.init.xavier_uniform_(self.feature_extractor[3].weight)
        torch.nn.init.xavier_uniform_(self.feature_extractor[6].weight)
        torch.nn.init.xavier_uniform_(self.classifier[0].weight)
        self.optimizer = torch.optim.Adam([{'params': self.feature_extractor.parameters()},
                                           {'params': self.classifier.parameters()}],
                                          lr=lr, amsgrad=True)

        self.crossLoss = nn.CrossEntropyLoss()
        self.iteration = 0
        self.best_iteration = 0
        self.loss_count = []
        self.loss_count_valid = []
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
            for i in range(self.n_classes):
                s_index = torch.nonzero(y == i).squeeze()
                index_random = torch.randperm(n=s_index.shape[0], generator=self.gen)
                s_ind = s_index[index_random]
                index = torch.cat((index, s_ind[0:vector[i].item()]), 0)
            index = index.type(torch.long)
            index = index.view(-1)
            index_rand = torch.randperm(len(index), generator=self.gen)
            index = index[index_rand]
            X_minibatch, y_minibatch = X[index], y[index]
            return X_minibatch.float(), y_minibatch.long(), index
        else:
            index = torch.randperm(len(X), generator=self.gen)
            index = index[:self.batchsize]
            X_target_minibatch = torch.tensor(X[index])
            return X_target_minibatch.float(), y, index

    def g(self, x):

        return self.feature_extractor(x)

    def f(self, features_conv):

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
        self.optimizer.step()

    def compute_total_loss(self, X_source, y_source, X_target, training=True):
        logprobas_source, conv_features_source = self.forward(X_source.transpose(1, 2))
        classif_loss = self.crossLoss(logprobas_source, y_source)
        if training:
            self.loss_count.append(classif_loss.item())
        return classif_loss, logprobas_source

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
        n_iter_avg = 10000
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
            if self.iteration % validation_step == 0:
                print(self.iteration)
                if self.saving:
                    names = ["loss_train.npy", "loss_valid.npy",
                             "acc_source.npy", "acc_target.npy"]
                    files = [self.loss_count, self.loss_count_valid,
                             self.acc_source, self.acc_target]
                    self.save_stuff(names=names, files=files)
        if self.saving:
            names = ["loss_train.npy", "loss_valid.npy",
                     "acc_source.npy", "acc_target.npy"]
            files = [self.loss_count, self.loss_count_valid,
                     self.acc_source, self.acc_target]
            self.save_stuff(names=names, files=files)

    def fit(self, X_source, y_source, X_target, X_source_valid, y_source_valid, X_target_valid, y_target_valid,
            max_iter, validation_step=1000):
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
        _, count_classes = torch.unique(self.y_source_, return_counts=True)
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
        for cl in range(0, self.n_classes):
            cl_bs = torch.round(self.batchsize * count_classes[cl] / torch.sum(count_classes))
            if cl_bs <= 1:
                cl_bs += 1
            sample_vec[cl] = cl_bs
        #  print(torch.argmax(sample_vec).item()[0], torch.argmax(sample_vec).shape)
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

        if max_iter > 0:
            self.iteration = 0
            self.fit_several_epochs(max_iter=max_iter, validation_step=validation_step)

    def new_iteration(self):
        self.iteration += 1

    def supervised_validation_step(self, verbose_step=1):

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
            if self.saving:
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
        return 100. * correct / len(labels)

    def latent_space(self, inputs, labels, inputs_target, labels_target, domain="target"):

        with torch.no_grad():
            self.eval()

            inputs = torch.tensor(inputs).type(torch.float)
            labels = torch.tensor(labels)
            inputs_target = torch.tensor(inputs_target).type(torch.float)
            labels_target = torch.tensor(labels_target)
            b_target_padded = torch.zeros(size=(256, 72, 10))
            if self.CUDA_train:
                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                    labels = labels.cuda()
                    inputs_target = inputs_target.cuda()
                    labels_target = labels_target.cuda()
                    b_target_padded = b_target_padded.cuda()

            _, count_classes = torch.unique(labels, return_counts=True)
            sample_vec = torch.zeros(size=(self.n_classes,))
            for cl in range(0, self.n_classes):
                cl_bs = torch.round(self.batchsize * count_classes[cl] / torch.sum(count_classes))
                if cl_bs <= 1:
                    cl_bs += 1
                sample_vec[cl] = cl_bs
            while sample_vec.sum() > self.batchsize:
                sample_vec[torch.argmax(sample_vec)] -= 1
            while sample_vec.sum() < self.batchsize:
                sample_vec[torch.argmin(sample_vec)] += 1
            self.sample_vec_train = sample_vec.type(torch.int)

            b_source, l_source, i_source = self.mini_batch_class_balanced(X=inputs, y=labels, vector=self.sample_vec_train)
            b_target, l_target, i_target = self.mini_batch_class_balanced(X=inputs_target, vector="pamplemousse")

            b_target_padded[:, :58] = b_target
            l_target = labels_target[i_target]

            out, out_cnn = self.forward(b_source.transpose(1, 2))
            out_target, out_cnn_target = self.forward(b_target_padded.transpose(1, 2))
            if self.saving:
                names = [domain + "_source_conv.npy", domain + "_target_conv.npy",
                         domain + "_source.npy", domain + "_target.npy"]
                files = [out_cnn.cpu(), out_cnn_target.cpu(), l_source.cpu(), l_target.cpu()]
                self.save_stuff(files=files, names=names)

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


if __name__ == '__main__':

    np.random.seed(300)
    torch.manual_seed(300)

    parser = argparse.ArgumentParser()
    parser.add_argument('-p1', '--pair1', type=int, help='The first half of the pair')
    parser.add_argument('-p2', '--pair2', type=int, help='The second half of the pair')
    parser.add_argument('-data', '--dataset', type=str, help="Which dataset to take between HAR and TarnBZH")
    parser.add_argument('-lr', "--learning_rate", type=float, help="The learning rate")
    parser.add_argument('-p', "--path", type=str)
    parser.add_argument('-e', "--epochs", type=int)
    parser.add_argument('-c', "--epochs_cnn", type=int)
    parser.add_argument('-n', "--name", type=str)

    bn_affine = False
    args, _ = parser.parse_known_args()

    def to_onehot(y):
        n_values = np.max(y) + 1
        return np.eye(n_values)[y]

    def from_numpy_to_torch(filename, float_or_long=True):
        data = np.load(filename)
        data_t = torch.from_numpy(data)
        if float_or_long:
            data_t = data_t.type(torch.float)
        else:
            data_t = data_t.type(torch.long)
        return data_t

    for paireee in [[14, 19]]:  # [[9, 18], [18, 23], [6, 23]]:  # [[2, 11], [7, 13], [12, 16], [12, 18]]:
        """pair1 = pair[0]
        pair2 = pair[1]
        chan = 9
        n_classes = 6
        source = pair1
        target = pair2
        # Train source dataset
        print('/share/home/fpainblanc/MAD-CNN/numpy_data/ucihar_' + str(source) + 'train.npy')
        train_source = from_numpy_to_torch('/share/home/fpainblanc/MAD-CNN/numpy_data/ucihar_' + str(source) +
                                           'train.npy')
        train_source_label = from_numpy_to_torch('/share/home/fpainblanc/MAD-CNN/numpy_data/ucihar_' + str(source) +
                                                 'train_labels.npy', float_or_long=False)
        # Valid source dataset
        valid_source = from_numpy_to_torch('/share/home/fpainblanc/MAD-CNN/numpy_data/ucihar_' + str(source) +
                                           'valid.npy')
        valid_source_label = from_numpy_to_torch('/share/home/fpainblanc/MAD-CNN/numpy_data/ucihar_' + str(source) +
                                                 'valid_labels.npy', float_or_long=False)
        # Test source dataset
        test_source = from_numpy_to_torch('/share/home/fpainblanc/MAD-CNN/numpy_data/ucihar_' + str(source) +
                                          'test.npy')
        test_source_label = from_numpy_to_torch('/share/home/fpainblanc/MAD-CNN/numpy_data/ucihar_' + str(source) +
                                                'test_labels.npy', float_or_long=False)
        # Train target dataset (no labels)
        train_target = from_numpy_to_torch('/share/home/fpainblanc/MAD-CNN/numpy_data/ucihar_' + str(target) +
                                           'train.npy')
        train_target_label = from_numpy_to_torch('/share/home/fpainblanc/MAD-CNN/numpy_data/ucihar_' + str(target) +
                                                 'train_labels.npy', float_or_long=False)
        # Valid source dataset
        valid_target = from_numpy_to_torch('/share/home/fpainblanc/MAD-CNN/numpy_data/ucihar_' + str(target) +
                                           'valid.npy')
        valid_target_label = from_numpy_to_torch('/share/home/fpainblanc/MAD-CNN/numpy_data/ucihar_' + str(target) +
                                                 'valid_labels.npy', float_or_long=False)
        # Test target dataset
        test_target = from_numpy_to_torch('/share/home/fpainblanc/MAD-CNN/numpy_data/ucihar_' + str(target) +
                                          'test.npy')
        test_target_label = from_numpy_to_torch('/share/home/fpainblanc/MAD-CNN/numpy_data/ucihar_' + str(target) +
                                                'test_labels.npy', float_or_long=False)"""
        if args.dataset == "TARNBZH":
            # Train source dataset
            n_classes = 6
            chan = 10
            if True:
                path = '/share/home/fpainblanc/MAD-CNN/numpy_data/'
            else:
                path = "/home/adr2.local/painblanc_f/Desktop/bzh_datasets/"
            print(path)
            train_source = from_numpy_to_torch(path + 'tarnbzh_1train.npy')
            train_source_label = from_numpy_to_torch(path + 'tarnbzh_1train_labels.npy', float_or_long=False)

            valid_source = from_numpy_to_torch(path + 'tarnbzh_1valid.npy')
            valid_source_label = from_numpy_to_torch(path + 'tarnbzh_1valid_labels.npy', float_or_long=False)

            test_source = from_numpy_to_torch(path + 'tarnbzh_1test.npy')
            test_source_label = from_numpy_to_torch(path + 'tarnbzh_1test_labels.npy', float_or_long=False)

            train_target = from_numpy_to_torch(path + 'tarnbzh_2train.npy')
            train_target_label = from_numpy_to_torch(path + 'tarnbzh_2train_labels.npy', float_or_long=False)

            valid_target = from_numpy_to_torch(path + 'tarnbzh_2valid.npy')
            valid_target_label = from_numpy_to_torch(path + 'tarnbzh_2valid_labels.npy', float_or_long=False)

            test_target = from_numpy_to_torch(path + 'tarnbzh_2test.npy')
            test_target_label = from_numpy_to_torch(path + 'tarnbzh_2test_labels.npy', float_or_long=False)
        if args.dataset == "TARNBZH5K":
            print(args.dataset)
            # Train source dataset
            source = str(args.pair1)
            target = str(args.pair2)

            n_classes = 5
            if source == "11":
                n_classes = 5
            chan = 10
            if args.path == "False":
                path = '/home/adr2.local/painblanc_f/codats-master/datas/numpy_data/tarnbzh5k_'
            if args.path == "True":
                path = '/share/home/fpainblanc/MAD-CNN/numpy_data/tarnbzh5k_'

            train_source = from_numpy_to_torch(path + source + 'train.npy')
            train_source_label = from_numpy_to_torch(path + source + 'train_labels.npy', float_or_long=False)

            valid_source = from_numpy_to_torch(path + source + 'valid.npy')
            valid_source_label = from_numpy_to_torch(path + source + 'valid_labels.npy', float_or_long=False)

            test_source = from_numpy_to_torch(path + source + 'test.npy')
            test_source_label = from_numpy_to_torch(path + source + 'test_labels.npy', float_or_long=False)

            train_target = from_numpy_to_torch(path + target + 'train.npy')
            train_target_label = from_numpy_to_torch(path + target + 'train_labels.npy', float_or_long=False)

            valid_target = from_numpy_to_torch(path + target + 'valid.npy')
            valid_target_label = from_numpy_to_torch(path + target + 'valid_labels.npy', float_or_long=False)

            test_target = from_numpy_to_torch(path + target + 'test.npy')
            test_target_label = from_numpy_to_torch(path + target + 'test_labels.npy', float_or_long=False)
        if args.dataset == "REMOTES":
            # Train source dataset
            if args.pair1 is None:
                source = "1"
                target = "2"
            else:
                source = str(args.pair1)
                target = str(args.pair2)

            n_classes = 8
            chan = 10
            if args.path == "False":
                path = '/home/adr2.local/painblanc_f/Desktop/bzh_datasets/remotes_'
            if args.path == "True":
                path = '/share/home/fpainblanc/MAD-CNN/numpy_data/remotes_'

            train_source = from_numpy_to_torch(path + source + 'train.npy')
            train_source_label = from_numpy_to_torch(path + source + 'train_labels.npy', float_or_long=False)

            valid_source = from_numpy_to_torch(path + source + 'valid.npy')
            valid_source_label = from_numpy_to_torch(path + source + 'valid_labels.npy', float_or_long=False)

            test_source = from_numpy_to_torch(path + source + 'test.npy')
            test_source_label = from_numpy_to_torch(path + source + 'test_labels.npy', float_or_long=False)

            train_target = from_numpy_to_torch(path + target + 'train.npy')
            train_target_label = from_numpy_to_torch(path + target + 'train_labels.npy', float_or_long=False)

            valid_target = from_numpy_to_torch(path + target + 'valid.npy')
            valid_target_label = from_numpy_to_torch(path + target + 'valid_labels.npy', float_or_long=False)

            test_target = from_numpy_to_torch(path + target + 'test.npy')
            test_target_label = from_numpy_to_torch(path + target + 'test_labels.npy', float_or_long=False)
            print(np.unique(train_target_label))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            accuracy_tot = np.empty(shape=(5, 3))
            paires = [[13, 11], [13, 12], [13, 14], [11, 13], [11, 12]]
            names = ["Dan2Bzh_eq", "Dan2Tarn_eq", "Dan2Aus_eq", "Bzh2Dan_eq", "Bzh2Tarn_eq"]
            p_index = 0
            for pair in range(0, 5):
                if args.dataset == "REMOTES7CL":
                    # Train source dataset
                    source = str(paires[pair][0])
                    target = str(paires[pair][1])

                    n_classes = 7
                    chan = 10
                    if args.path == "False":
                        path = '/home/adr2.local/painblanc_f/codats-master/datas/numpy_data/remotes7cl_'
                    if args.path == "True":
                        path = '/share/home/fpainblanc/MAD-CNN/numpy_data/remotes7cl_'

                    train_source = from_numpy_to_torch(path + source + 'train.npy')
                    train_source_label = from_numpy_to_torch(path + source + 'train_labels.npy', float_or_long=False)

                    valid_source = from_numpy_to_torch(path + source + 'valid.npy')
                    valid_source_label = from_numpy_to_torch(path + source + 'valid_labels.npy', float_or_long=False)

                    test_source = from_numpy_to_torch(path + source + 'test.npy')
                    test_source_label = from_numpy_to_torch(path + source + 'test_labels.npy', float_or_long=False)

                    train_target = from_numpy_to_torch(path + target + 'train.npy')
                    train_target_label = from_numpy_to_torch(path + target + 'train_labels.npy', float_or_long=False)

                    valid_target = from_numpy_to_torch(path + target + 'valid.npy')
                    valid_target_label = from_numpy_to_torch(path + target + 'valid_labels.npy', float_or_long=False)

                    test_target = from_numpy_to_torch(path + target + 'test.npy')
                    test_target_label = from_numpy_to_torch(path + target + 'test_labels.npy', float_or_long=False)
                feature_extractor = nn.Sequential(
                    nn.Conv1d(in_channels=chan, out_channels=128, kernel_size=8, stride=1, padding="same", bias=False),
                    nn.BatchNorm1d(num_features=128, affine=bn_affine),
                    nn.ReLU(),

                    nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, stride=1, padding="same", bias=False),
                    nn.BatchNorm1d(num_features=256, affine=bn_affine),
                    nn.ReLU(),

                    nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding="same", bias=False),
                    nn.BatchNorm1d(num_features=128, affine=bn_affine),
                    nn.ReLU()
                )

                classifier = nn.Sequential(nn.Linear(128, n_classes))  # /!\ Does not include softmax activation
                r_index = 0
                for rep in ["_rep1", "_rep2_", "_rep3_"]:
                    name_model = names[pair] + rep
                    #  name = "/share/home/fpainblanc/CNNMAD/REMOTES7CL/Baseline/" + name_model
                    name = "/share/home/fpainblanc/CNNMAD/REMOTES7CL/Baseline/" + name_model
                    CNN_mod = CNNMAD(name=name, batchsize=256, feature_extractor=feature_extractor,
                                     classifier=classifier, n_classes=n_classes,
                                     supervised_validation=False, lr=0.0001, saving=True, CUDA_train=True)
                    """CNN_mod.fit(X_source=train_source, y_source=train_source_label,
                                X_target=train_source, X_source_valid=valid_source,
                                y_source_valid=valid_source_label, X_target_valid=valid_source,
                                y_target_valid=valid_source_label, max_iter=args.epochs, validation_step=500)"""
                    """CNN_mod.fit(X_source=train_source, y_source=train_source_label,
                                X_target=train_source, X_source_valid=valid_source,
                                y_source_valid=valid_source_label, X_target_valid=valid_source,
                                y_target_valid=valid_source_label, max_iter=args.epochs, validation_step=5000)"""
                    if os.path.exists(name + "30000.pt"):
                        CNN_mod.load_state_dict(torch.load(name + "30000.pt"))
                    else:
                        CNN_mod.load_state_dict(torch.load(name + "30000best.pt"))
                    #  CNN_mod.evaluate(inputs=test_source, labels=test_source_label)
                    accuracy = CNN_mod.evaluate(inputs=test_target, labels=test_target_label)
                    accuracy_tot[p_index, r_index] = accuracy
                    r_index += 1
                    #  CNN_mod.latent_space(test_source, test_source_label, test_target, test_target_label, domain="test")

                    """name = "/share/home/fpainblanc/CNNMAD/REMOTES/baseline/" + str(learning_rate) + "/trainOnSourceD2B"
                    CNN_mod = CNNMAD(name=name, batchsize=256, feature_extractor=feature_extractor,
                                     classifier=classifier, n_classes=n_classes, lamb=1., alpha=0.0, beta=0.0,
                                     MAD_class=True, supervised_validation=False, lr=learning_rate, saving=True,
                                     CUDA_train=True)
                    CNN_mod.fit(X_source=train_source, y_source=train_source_label,
                                X_target=train_source, X_source_valid=valid_source,
                                y_source_valid=valid_source_label, X_target_valid=valid_source,
                                y_target_valid=valid_source_label, max_iter=0,
                                n_iter_pretrain_cnn=30000, validation_step=30000)
                    CNN_mod.evaluate(inputs=test_target, labels=test_target_label)"""
                p_index += 1
            np.save("REMOTES7CL_acc/remotes_s2t.npy", accuracy_tot)
