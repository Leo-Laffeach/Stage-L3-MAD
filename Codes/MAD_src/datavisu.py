import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE, MDS
import os


class PlotTraining:
    def __init__(self, name, iteration, classe, alpha, beta, show=False):
        self.name = name
        self.iteration = iteration
        self.classe = classe
        self.alpha = alpha
        self.beta = beta
        self.show = show

        """self.valid_source_conv = np.load(self.name + str(self.iteration) + "validConv_source.npy")
        self.valid_target_conv = np.load(self.name + str(self.iteration) + "validConv_target.npy")"""

    @staticmethod
    def to_onehot(y):
        n_values = np.max(y) + 1
        return np.eye(n_values)[y]

    def all_loss_plot(self, name=None):
        total_loss = np.load(self.name + str(self.iteration) + "loss_train.npy")
        """index = np.arange(total_loss.shape[0])
        total_loss = total_loss[index % 2 == 0]"""
        if self.alpha > 0:
            alpha_loss = np.load(self.name + str(self.iteration) + "loss_alpha.npy")
            """print(alpha_loss.shape)
            index = np.arange(alpha_loss.shape[0])
            alpha_loss = alpha_loss[index % 2 == 0]"""
        else:
            alpha_loss = np.zeros(shape=total_loss.shape)
        if self.beta > 0:
            beta_loss = np.load(self.name + str(self.iteration) + "loss_beta.npy")
            if beta_loss.shape[0] > total_loss.shape[0]:
                index = np.arange(beta_loss.shape[0])
                beta_loss = beta_loss[index % 2 == 0]
        else:
            beta_loss = np.zeros(shape=total_loss.shape)
        print(alpha_loss.shape, beta_loss.shape, total_loss.shape)
        plt.plot(alpha_loss, label="Alpha Loss")
        plt.plot(beta_loss, label="Beta Loss")
        plt.plot(total_loss, label="Classification Loss")
        plt.xlabel('Epochs')
        plt.ylabel('Losses')
        plt.title(name)
        plt.legend()
        plt.savefig(self.name + "all_losses.png")
        if self.show:
            plt.show()
        # plt.clf()

    def valid_loss_plot(self, name=None):

        loss_valid = np.load(self.name + str(self.iteration) + "loss_valid.npy")
        plt.plot(loss_valid, label="Validation loss on source")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title(name)
        plt.savefig(self.name + "valid_loss.png")
        if self.show:
            plt.show()
        # plt.clf()

    def weighted_loss_plot(self, name=None):

        total_loss = np.load(self.name + str(self.iteration) + "loss_train.npy")
        index = np.arange(total_loss.shape[0])
        total_loss = total_loss[index % 2 is True]

        if self.alpha > 0:
            alpha_loss = np.load(self.name + str(self.iteration) + "loss_alpha.npy", allow_pickle=True)
            """index = np.arange(alpha_loss.shape[0])
            alpha_loss = alpha_loss[index % 2 is True]"""
        else:
            alpha_loss = np.zeros(shape=total_loss.shape)
        if self.beta > 0:
            beta_loss = np.load(self.name + str(self.iteration) + "loss_beta.npy", allow_pickle=True)
            """index = np.arange(beta_loss.shape[0])
            beta_loss = beta_loss[index % 2 is True]"""
            """if beta_loss.shape[0] > total_loss.shape[0]:
                index = np.arange(beta_loss.shape[0])
                beta_loss = beta_loss[index % 2 is True]"""
        else:
            beta_loss = np.zeros(shape=total_loss.shape)

        plt.plot(self.alpha * alpha_loss, label="Alpha Loss")
        plt.plot(self.beta * beta_loss, label="Beta Loss")
        plt.plot(total_loss, label="Classification Loss")
        plt.xlabel('Epochs')
        plt.ylabel('Losses')
        plt.title(name)
        plt.legend()
        plt.savefig(self.name + "weighted_losses.png")
        if self.show:
            plt.show()
        # plt.clf()

    def validation_plot(self, name=None, mean=None):

        # valid_source_accuracy = np.load(self.name + str(self.iteration) + "acc_source.npy")
        valid_target_accuracy = np.load(self.name + str(self.iteration) + "acc_target.npy")
        # valid_target_accuracy = np.load(self.name + str(self.iteration) + 'total_acc.npy')
        # plt.plot(valid_source_accuracy, label="Validation source accuracy")
        plt.plot(valid_target_accuracy, label="Validation target accuracy")
        if mean is not None:
            plt.axhline(mean, xmin=0, xmax=self.iteration, color="red", label="CoDATS")
        # plt.plot(np.arange(0, valid_source_accuracy.shape[0], 20), valid_target_accuracy, label="Validation target accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.title(name)
        plt.legend()
        plt.savefig(self.name + "accuracy_validation.png")
        if self.show:
            plt.show()
        # plt.clf()

    def OT_train_plot(self, iteration=None):

        if iteration is None:
            iteration = self.iteration
        train_source_label = np.load(self.name + str(iteration) + "trainlabels_source.npy")
        train_target_label = np.load(self.name + str(iteration) + "trainlabels_target.npy", allow_pickle=True)
        OT_train = np.load(self.name + str(iteration) + "trainOT_forward_MAD.npy")

        yt_onehot = self.to_onehot(train_source_label)
        y_pred = np.argmax(np.dot(OT_train.T, yt_onehot), axis=1)
        print('accuracy basic_OT ={}'.format(np.mean(y_pred == train_target_label)))

        sort_source = np.argsort(train_source_label)
        sort_target = np.argsort(train_target_label)
        OT_clas = OT_train[sort_source]
        OT_classed = OT_clas[:, sort_target]

        plt.imshow(OT_classed)
        plt.title("Training datasets OT matrix ordered by classes")
        plt.xlabel('Target domain')
        plt.ylabel("Source domain")
        plt.savefig(self.name + "OT_train.png")
        if self.show:
            plt.show()
        # plt.clf()

    def OT_test_plot(self, iteration=None):

        if iteration is None:
            iteration = self.iteration
        train_source_label = np.load(self.name + str(iteration) + "testlabels_source.npy")
        train_target_label = np.load(self.name + str(iteration) + "testlabels_target.npy", allow_pickle=True)
        OT_train = np.load(self.name + str(iteration) + "testOT_forward_MAD.npy")

        yt_onehot = self.to_onehot(train_source_label)
        y_pred = np.argmax(np.dot(OT_train.T, yt_onehot), axis=1)
        print('accuracy test basic_OT ={}'.format(np.mean(y_pred == train_target_label)))

        sort_source = np.argsort(train_source_label)
        sort_target = np.argsort(train_target_label)
        OT_clas = OT_train[sort_source]
        OT_classed = OT_clas[:, sort_target]

        plt.imshow(OT_classed)
        plt.title("Training datasets OT matrix ordered by classes")
        plt.xlabel('Target domain')
        plt.ylabel("Source domain")
        plt.savefig(self.name + "OT_train.png")
        if self.show:
            plt.show()
        # plt.clf()

    def OT_valid_plot(self, iteration=None):
        if iteration is None:
            iteration = self.iteration
        """OT_valid = np.load(self.name + str(iteration) + "validOT_forward_MAD.npy")
        valid_source_label = np.load(self.name + str(iteration) + "validlabels_source.npy")
        valid_target_label = np.load(self.name + str(iteration) + "validlabels_target.npy")"""
        OT_valid = np.load("test_classif0OT_matrice.npy")
        valid_source_label = np.load("test_classif0label_source.npy")
        valid_target_label = np.load('test_classif0label_target.npy')

        yt_onehot = self.to_onehot(valid_source_label)
        y_pred = np.argmax(np.dot(OT_valid.T, yt_onehot), axis=1)
        print('accuracy valid basic_OT ={}'.format(np.mean(y_pred == valid_target_label)))

        sort_source = np.argsort(valid_source_label)
        sort_target = np.argsort(valid_target_label)
        OT_clas = OT_valid[sort_source]
        OT_classed = OT_clas[:, sort_target]

        plt.imshow(OT_classed)
        plt.title("Training datasets OT matrix ordered by classes at iter" + str(iteration))
        plt.xlabel('Target domain')
        plt.ylabel("Source domain")
        plt.savefig(self.name + "OT_validation.png")
        if self.show:
            plt.show()
        # plt.clf()

    def la_tante_rep(self, classe, iteration=None, name="test"):
        if iteration is None:
            iteration = self.iteration
        """train_source_conv = np.load(self.name + str(iteration) + name + "Conv_source.npy")
        train_target_conv = np.load(self.name + str(iteration) + name + "Conv_target.npy")
        train_source_label = np.load(self.name + str(iteration) + name + "labels_source.npy")
        train_target_label = np.load(self.name + str(iteration) + name + "labels_target.npy", allow_pickle=True)
        OT_train = np.load(self.name + str(iteration) + name + "OT_forward_MAD.npy")
        train_OT_cost = np.load(self.name + str(iteration) + name + "OT_Cost_forward_MAD.npy")
        train_DTW = np.load(self.name + str(iteration) + name + "DTW_forward_MAD.npy")
        pred = np.load(self.name + str(iteration) + name + "pred_forward.npy")"""
        train_source_label = np.load("test_classif0label_source.npy")
        train_target_label = np.load("test_classif0label_target.npy")
        OT_train = np.load("test_classif0OT_matrice.npy")
        train_source_conv = np.load("test_classif0source.npy")
        train_target_conv = np.load("test_classif0target.npy")
        train_OT_cost = np.load("test_classif0Cost.npy")
        train_DTW = np.load("test_classif0DTW.npy")

        """train_source_conv = np.load(self.name + str(self.iteration) + "trainConv_source.npy")
        train_target_conv = np.load(self.name + str(self.iteration) + "trainConv_target.npy")
        train_source_label = np.load(self.name + str(self.iteration) + "trainlabels_source.npy")
        train_target_label = np.load(self.name + str(self.iteration) + "trainlabels_target.npy", allow_pickle=True)
        OT_train = np.load(self.name + str(self.iteration) + "trainOT_forward_MAD.npy")
        train_OT_cost = np.load(self.name + str(self.iteration) + "trainOT_Cost_forward_MAD.npy")
        train_DTW = np.load(self.name + str(self.iteration) + "trainDTW_forward_MAD.npy")
        pred = np.load(self.name + str(self.iteration) + "trainpred_forward.npy")"""

        index_source = np.argsort(train_source_label)
        index_target = np.argsort(train_target_label)
        train_OT_cost_sort = train_OT_cost[index_source]
        train_OT_cost_sorted = train_OT_cost_sort[:, index_target]
        train_source_label_sorted = train_source_label[index_source]
        train_target_label_sorted = train_target_label[index_target]
        train_source_sorted = train_source_conv[index_source]
        train_target_sorted = train_target_conv[index_target]
        OT_train_sort = OT_train[index_source]
        OT_train_sorted = OT_train_sort[:, index_target]
        """pred_sorted = pred[index_target]
        pred_sorted = pred_sorted.squeeze(-1)
        yt_onehot = self.to_onehot(train_source_label_sorted)
        y_pred = np.argmax(np.dot(OT_train_sorted.T, yt_onehot), axis=1)
        print('accuracy basic_OT ={}'.format(np.mean(y_pred == train_target_label_sorted)))"""
        from sklearn.metrics.pairwise import euclidean_distances
        source_dist = euclidean_distances(train_source_sorted.reshape(train_source_sorted.shape[0], -1), squared=True)
        source_dist /= train_source_sorted.shape[1]
        target_dist = euclidean_distances(train_target_sorted.reshape(train_target_sorted.shape[0], -1), squared=True)
        target_dist /= train_target_sorted.shape[1]
        source_target_cost_w = np.empty(shape=(train_source_sorted.shape[0], train_target_sorted.shape[0]))

        for c in range(0, classe):
            source_target_cost_w[train_source_label_sorted == c] = (train_OT_cost_sorted[train_source_label_sorted == c]
                                                                    * train_source_sorted.shape[2]) / train_DTW[c].sum()

        max_OT_train = OT_train_sorted * (OT_train_sorted >= np.sort(OT_train_sorted, axis=1)[:, [-1]])
        """plt.subplot(121)
        plt.imshow(OT_train_sorted)
        plt.subplot(122)
        plt.imshow(max_OT_train)
        # plt.show()
        plt.clf()"""
        indice_matrix = np.nonzero(max_OT_train)
        ind_row = indice_matrix[0]
        ind_col = indice_matrix[1]
        Cost1 = np.concatenate((source_dist, source_target_cost_w), axis=1)
        Cost2 = np.concatenate((source_target_cost_w.T, target_dist), axis=1)
        Cost = np.concatenate((Cost1, Cost2), axis=0)
        transform = MDS(n_components=2, random_state=1, dissimilarity="precomputed").fit_transform(Cost)
        source_transform = transform[:train_source_sorted.shape[0]]
        target_transform = transform[train_source_sorted.shape[0]:]
        color_index = ['blue', 'orange', 'red', 'green', 'pink', 'purple']
        """for pair in range(0, len(ind_row)):
            if train_target_label_sorted[ind_col[pair]] != pred_sorted[ind_col[pair]]:
                if pair == 0:
                    plt.plot([source_transform[ind_row[pair], 0], target_transform[ind_col[pair], 0]],
                             [source_transform[ind_row[pair], 1], target_transform[ind_col[pair], 1]],
                             color="black", linestyle='--', linewidth=0.3, label="Wrongly classified")
                else:
                    plt.plot([source_transform[ind_row[pair], 0], target_transform[ind_col[pair], 0]],
                             [source_transform[ind_row[pair], 1], target_transform[ind_col[pair], 1]],
                             color="black", linestyle='--', linewidth=0.3)"""
        plt.title("Projection on iteration" + str(iteration) + "on " + name)
        plt.savefig(self.name + "latent_representation_eq_datasets_correct.png")
        for c in range(0, classe):
            plt.scatter(source_transform[train_source_label_sorted == c, 0],
                        source_transform[train_source_label_sorted == c, 1],
                        marker='*', label="source MAD", c=color_index[c])
            plt.scatter(target_transform[train_target_label_sorted == c, 0],
                        target_transform[train_target_label_sorted == c, 1],
                        marker='+', label="target MAD", c=color_index[c])
            if c == 0:
                plt.legend()
        if self.show:
            plt.show()

    def plot_DTW(self):

        train_DTW = np.load(self.name + str(self.iteration) + "trainDTW_forward_MAD.npy")
        cols = int(len(train_DTW) / 2)
        rows = 2
        indice = 0
        fig, axes = plt.subplots(rows, cols, figsize=(16, 12))
        for i in range(0, rows):
            for j in range(0, cols):
                axes[i][j].imshow(train_DTW[indice])
                indice += 1
        plt.show()
        # plt.clf()


if __name__ == '__main__':
    # for a in [0.0, 1.0, 0.1, 0.01, 0.001, 0.0001, 0.00001]:
    for a in [0.01]:
        # for b in [1.0, 0.1, 0.01, 0.0]:
        for b in [1.0]:
            for i in [0]:
                lr = 0.001
                # path = "TARNBZH_HYPERPARAM/" + str(a) + "/" + str(b) + "/rep1"
                path = ""
                hebo = True

                """rep1 = np.load("UCIHAR_rep/0.01/0.1/rep11000total_acc.npy")
                rep2 = np.load("UCIHAR_rep/0.01/0.1/rep21000total_acc.npy")
                rep3 = np.load("UCIHAR_rep/0.01/0.1/rep31000total_acc.npy")
                rep4 = np.load("UCIHAR_rep/0.01/0.1/rep41000total_acc.npy")

                plt.plot(rep1, label='Accuracy on validation set')
                plt.plot(rep2)
                plt.plot(rep3)
                plt.plot(rep4)
                plt.title("HAR pair 14-19 alpha=0.01, beta=0.1, four repetitions")
                plt.legend()
                plt.show()"""
                it = 500
                plott = PlotTraining(name=path, iteration=i, classe=6, alpha=a, beta=b, show=False)
                # plott.OT_train_plot(iteration=it)
                # plott.OT_valid_plot(iteration=it)
                # plott.OT_test_plot(iteration=it)
                # plott.all_loss_plot("Losses")
                if hebo:
                    plt.suptitle("a=" + str(a) + "b=" + str(b) + " " + str(i) +
                                 "iterations, lr=" + str(lr) + "CoDATS network, tarn to bzh")

                    plt.subplot(221)
                    # plott.validation_plot("Accuracy plot")
                    # plott.all_loss_plot('Losses')
                    # plott.la_tante_rep(classe=6, iteration=1000, name="test")
                    plt.subplot(222)
                    # plott.valid_loss_plot("Validation loss on source domain")
                    # plott = PlotTraining(name=path, iteration=1000, classe=6, alpha=a, beta=b, show=False)
                    # plott.la_tante_rep(classe=6)
                    # plott.OT_valid_plot(iteration=5000)
                    # plott.validation_plot("Accuracy plot", mean=80.00)
                    # plott.plot_DTW()
                    plt.subplot(223)
                    # plott = PlotTraining(name=path, iteration=3000, classe=6, alpha=a, beta=b, show=False)
                    plott.la_tante_rep(classe=6)
                    plt.subplot(224)
                    # plott = PlotTraining(name=path, iteration=5000, classe=6, alpha=a, beta=b, show=False)
                    # plott.la_tante_rep(classe=6, name="test_eq")
                    plott.OT_valid_plot()
                    plt.tight_layout()
                    plt.show()
