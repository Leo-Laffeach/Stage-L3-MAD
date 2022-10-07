from datavisu import PlotTraining
import numpy as np
from tslearn.datasets import CachedDatasets
import torch
import MADCNN
import MADonlyCNN
import warnings
import matplotlib.pyplot as plt

X, Y, _, _ = CachedDatasets().load_dataset(dataset_name='Trace')
X = X[np.where(Y == 1)]
Y = Y[np.where(Y == 1)] - 1

X_source = np.empty(shape=(X.shape[0], X.shape[1] + 30, 1))
X_source[:, :-30] = X
X_source[:, -30:] = np.repeat(repeats=30, a=X[:, -1]).reshape(X.shape[0], 30, 1)

X_target = np.empty(shape=(X.shape[0], X.shape[1] + 30, 1))
X_target[:, 30:] = X
X_target[:, :30] = np.repeat(repeats=30, a=X[:, 0]).reshape(X.shape[0], 30, 1)

X_source = torch.tensor(X_source, requires_grad=True)
X_target = torch.tensor(X_target)

X_target_clear = X_target

X_target = X_target + torch.normal(mean=0, std=1, size=X_target.shape)
X_target.requires_grad = True

Y = torch.tensor(Y).long()

loss = []
diff = []
diffAB = []
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    cnnmad = MADonlyCNN.CNNMAD(num_class=1, train_source_data=X_source, train_source_label=Y, reg=0.,
                               train_target_data=X_target, train_target_label=Y, channel=1, batchsize=30,
                               name="batch/batchnorm_affine", batchnorm=True, affine=True, reg_rep=True)
    cnnmad.fit(iterations=10000, validation=10000)
    cnnmad = MADonlyCNN.CNNMAD(num_class=1, train_source_data=X_source, train_source_label=Y, reg=0.,
                               train_target_data=X_target, train_target_label=Y, channel=1, batchsize=30,
                               name="batch/batchnorm_no_affine", batchnorm=True, affine=False, reg_rep=True)
    cnnmad.fit(iterations=10000, validation=10000)
    cnnmad = MADonlyCNN.CNNMAD(num_class=1, train_source_data=X_source, train_source_label=Y, reg=0.,
                               train_target_data=X_target, train_target_label=Y, channel=1, batchsize=30,
                               name="batch/no_batchnorm", batchnorm=False, affine=True, reg_rep=True)
    cnnmad.fit(iterations=10000, validation=10000)
    cnnmad = MADonlyCNN.CNNMAD(num_class=1, train_source_data=X_source, train_source_label=Y, reg=0.01,
                               train_target_data=X_target, train_target_label=Y, channel=1, batchsize=30,
                               name="batch/weight_affine", batchnorm=True, affine=True, reg_rep=False)
    cnnmad.fit(iterations=3000, validation=3000)
    cnnmad = MADonlyCNN.CNNMAD(num_class=1, train_source_data=X_source, train_source_label=Y, reg=0.01,
                               train_target_data=X_target, train_target_label=Y, channel=1, batchsize=30,
                               name="batch/latent_affine", batchnorm=True, affine=True, reg_rep=True)
    cnnmad.fit(iterations=3000, validation=3000)
    cnnmad = MADonlyCNN.CNNMAD(num_class=1, train_source_data=X_source, train_source_label=Y, reg=0.01,
                               train_target_data=X_target, train_target_label=Y, channel=1, batchsize=30,
                               name="batch/weight_no_affine", batchnorm=True, affine=False, reg_rep=False)
    cnnmad.fit(iterations=3000, validation=3000)
    cnnmad = MADonlyCNN.CNNMAD(num_class=1, train_source_data=X_source, train_source_label=Y, reg=0.01,
                               train_target_data=X_target, train_target_label=Y, channel=1, batchsize=30,
                               name="batch/latent_no_affine", batchnorm=True, affine=False, reg_rep=True)
    cnnmad.fit(iterations=3000, validation=3000)
"""plot_CNNMAD = PlotTraining(name="regul/alpha", iteration=1000, classe=1)

plot_CNNMAD.OT_train_plot()
plot_CNNMAD.OT_valid_plot()
plot_CNNMAD.latent_rep_train()"""


"""Check if while optimizing MAD(A, B') and updating B' it converges toward B
Does converge toward A in fact"""

"""for e in range(100):
    X_target.grad.data.zero_()
    cnnmad = MADCNN.CNNMAD(num_class=1, alpha=1., beta=0., lamb=0., train_source_data=X_source, train_source_label=Y,
                           train_target_data=X_target, channel=1, batchsize=30, name="")
    cnnmad.set_current_batchsize(X_source.shape[0])
    cnnmad.set_current_batchsize(X_source.shape[0], train=False)
    cnnmad.mad(out_conv_source=X_source, out_conv_target=X_target, labels_source=Y)
    loss_MAD = cnnmad.loss_CNN_MAD(labels_source=Y, out_conv_source=X_source, out_conv_target=X_target)

    loss_MAD.backward()
    with torch.no_grad():
        X_target += - 0.01 * X_target.grad
    diff.append(X_target.sum() - X_target_clear.sum())
    diffAB.append(X_target.sum() - X_source.sum())
    loss.append(loss_MAD)
np.save("AB_OT_1.npy", cnnmad._OT)

plt.plot(loss)
plt.title("Loss MAD on sum difference with and without noise")
plt.xlabel("Epoch")
plt.ylabel('Value of the loss')
plt.show()

plt.clf()
plt.plot(diff)
plt.title("Sum difference with and without noise")
plt.xlabel("Epoch")
plt.ylabel('Value of the difference')
plt.show()

plt.clf()
plt.plot(diffAB)
plt.title("Sum difference with noise and original dataset")
plt.xlabel("Epoch")
plt.ylabel('Value of the difference')
plt.show()"""

