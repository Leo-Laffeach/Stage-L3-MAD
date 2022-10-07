import torch.nn as nn
import torch
import numpy as np
from tslearn.datasets import CachedDatasets
import warnings
from MADCNN import CNNMAD

torch.manual_seed(200)
np.random.seed(200)

X, Y, _, _ = CachedDatasets().load_dataset(dataset_name='Trace')
X = X[np.where(Y == 1)]
Y = Y[np.where(Y == 1)] - 1

X_source = np.empty(shape=(X.shape[0], X.shape[1] + 30, 1))
X_source[:, :-30] = X
X_source[:, -30:] = np.repeat(repeats=30, a=X[:, -1]).reshape(X.shape[0], 30, 1)
y_source = torch.tensor(Y).long()

X_target = np.empty(shape=(X.shape[0], X.shape[1] + 30, 1))
X_target[:, 30:] = X
X_target[:, :30] = np.repeat(repeats=30, a=X[:, 0]).reshape(X.shape[0], 30, 1)

X_source = torch.tensor(X_source, requires_grad=True)
X_target = torch.tensor(X_target)

X_target_clear = torch.clone(X_target)

X_target = X_target + torch.normal(mean=0, std=1, size=X_target.shape)
X_target.requires_grad = True
ts_channels = X.shape[-1]
n_classes = 1  # len(set(Y))
bn_affine = False  # Set to False, otherwise, the latent representation might shrink to 0

feature_extractor = nn.Sequential(
    nn.Conv1d(in_channels=ts_channels, out_channels=128, kernel_size=8, stride=1, padding="same", bias=False),
    nn.BatchNorm1d(num_features=128, affine=bn_affine),
    nn.ReLU(),
    
    nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, stride=1, padding="same", bias=False),
    nn.BatchNorm1d(num_features=256, affine=bn_affine),
    nn.ReLU(),
    
    nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding="same", bias=False),
    nn.BatchNorm1d(num_features=128, affine=bn_affine),
    nn.ReLU()
)

classifier = nn.Sequential(
    nn.Linear(128, n_classes)
)  # /!\ Does not include softmax activation

full_model = CNNMAD(
    name="", batchsize=50,
    alpha=1e-1, beta=1e-1, lamb=1., 
    MAD_class=True, lr=1e-3, saving=True, CUDA_train=False,
    feature_extractor=feature_extractor, classifier=classifier, n_classes=n_classes
)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    full_model.fit(X_source=X_source, y_source=y_source, X_target=X_target, X_source_valid=X_source,
                   y_source_valid=y_source, X_target_valid=X_target, y_target_valid=y_source, max_iter=7,
                   validation_step=3)
