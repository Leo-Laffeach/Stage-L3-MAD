import numpy as np

import torch
import logging

from MAD_loss_ import MAD_loss

from utils import nanmean, MAE, RMSE

class OT_MAD():
    """
    'One parameter equals one imputed value' model
    Parameters
    ----------
        
    lr : float, default = 0.01
        Learning rate.

    opt: torch.nn.optim.Optimizer, default=torch.optim.Adam
        Optimizer class to use for fitting.
        
    max_iter : int, default=10
        Maximum number of round-robin cycles for imputation.

    niter : int, default=15
        Number of gradient updates for each model within a cycle.

    batchsize : int, defatul=128
        Size of the batches on which the sinkhorn divergence is evaluated.

    n_pairs : int, default=10
        Number of batch pairs used per gradient update.

    tol : float, default = 0.001
        Tolerance threshold for the stopping criterion.

    weight_decay : float, default = 1e-5
        L2 regularization magnitude.

    order : str, default="random"
        Order in which the variables are imputed.
        Valid values: {"random" or "increasing"}.

    unsymmetrize: bool, default=True
        If True, sample one batch with no missing 
        data in each pair during training.

    scaling: float, default=0.9
        Scaling parameter in Sinkhorn iterations
        c.f. geomloss' doc: "Allows you to specify the trade-off between
        speed (scaling < .4) and accuracy (scaling > .9)"


    """
    def __init__(self,
                 num_class,
                 MAD_class,
                 lr=1e-2, 
                 opt=torch.optim.RMSprop, 
                 niter=2000,
                 batchsize=128,
                 n_pairs=1,
                 noise=0.1,
                 scaling=.9 ):
        self.lr = lr
        self.opt = opt
        self.niter = niter
        self.batchsize = batchsize
        self.n_pairs = n_pairs
        self.noise = noise

        self.MAD_loss = MAD_loss(num_class,
                                 MAD_class,  
                                 target_prop=None,
                                 unbalanced=False, 
                                 un_reg=None, 
                                 un_mas=1.0)

    def fit_transform(self, target, source, labels_source,
                      verbose = True,
                      report_interval = 1, 
                      target_true = None, 
                      source_true = None, 
                      return_losses = False ):
        """
            Imputes missing data in source with target using MAD_loss

        Parameters
        ----------
        target : torch.FloatTensor or torch.cuda.DoubleTensor, shape (n, t, d)
            Contains non-missing and missing data at the indices given by the
            "mask" argument. Missing values can be arbitrarily assigned 
            (e.g. with NaNs).
            
        source : torch.FloatTensor or torch.cuda.DoubleTensor, shape (n, t, d)
            Contains non-missing and missing data at the indices given by the
            "mask" argument. Missing values can be arbitrarily assigned 
            (e.g. with NaNs).
        
        labels_source : 
            needed but not use.

        similarity_CE :

        verbose : bool, default=True
            If True, output loss to log during iterations.
            
        report_interval : int, default=1
            Interval between loss reports (if verbose).

        target_true: torch.FloatTensor or None, default=None
            Ground truth for the missing values. If provided, will output a 
            validation score during training. For debugging only.
        
        source_true: torch.FloatTensor or None, default=None
            Ground truth for the missing values. If provided, will output a 
            validation score during training. For debugging only.

        Returns
        -------
        target_filled: torch.FloatTensor or torch.cuda.FloatTensor
            Imputed missing data (plus unchanged non-missing data).
        
        source_filled: torch.FloatTensor or torch.cuda.FloatTensor
            Imputed missing data (plus unchanged non-missing data).

        """
        # target
        target = target.clone()
        n_target , t_target, d_target = target.shape
        td_target = t_target * d_target
        target = target.reshape(n_target, td_target)

        if target_true is not None:
            target_true = target_true.reshape(n_target, td_target)

        mask_target= torch.isnan(target).float()
        missing_pourcentage_target = mask_target.sum() / (n_target * td_target)

        # source
        source = source.clone()
        n_source, t_source, d_source = source.shape
        td_source = t_source * d_source 
        source = source.reshape(n_source, td_source)
        
        if source_true is not None:
            source_true = source_true.reshape(n_source, td_source)

        mask_source = torch.isnan(source).float()
        missing_pourcentage_source = mask_source.sum() / (n_source * td_source)

        if return_losses:
            losses = []
        
        if self.batchsize > n_target // 2 or self.batchsize > n_source // 2:
            n = min(n_target, n_source)
            e = int(np.log2(n // 2))
            self.batchsize = 2**e
            if verbose:
                logging.info(f"Batchsize larger that half size = {max(n_target, n_source) // 2}. Setting batchsize to {self.batchsize}.")

        # target
        imps_target = (self.noise * torch.randn(mask_target.shape).float() + nanmean(target, 0))[mask_target.bool()]
        imps_target.requires_grad = True
        #source
        imps_source = (self.noise * torch.randn(mask_source.shape).float() + nanmean(source, 0))[mask_source.bool()]
        imps_source.requires_grad = True
        
        optimizer = self.opt([imps_target, imps_source], lr=self.lr)

        if verbose:
            logging.info(f"batchsize = {self.batchsize}")
            logging.info(f"Missing pourcentage target = {missing_pourcentage_target}")
            logging.info(f"Missing pourcentage source = {missing_pourcentage_source}")

        if target_true is not None:
            maes_target = np.zeros(self.niter)
            rmses_target = np.zeros(self.niter)

        if source_true is not None:
            maes_source = np.zeros(self.niter)
            rmses_source = np.zeros(self.niter)
        for i in range(self.niter):
            
            target_filled = target.detach().clone()
            target_filled[mask_target.bool()] = imps_target

            source_filled = source.detach().clone()
            source_filled[mask_source.bool()] = imps_source

            loss = 0
            
            for _ in range(self.n_pairs):

                idx_target = np.random.choice(n_target, self.batchsize, replace=False)
                idx_source = np.random.choice(n_source, self.batchsize, replace=False)
    
                out_conv_target = target_filled.reshape(n_target, t_target, d_target)[idx_target]
                out_conv_source = source_filled.reshape(n_source, t_source, d_source)[idx_source]
    
                loss = loss + self.MAD_loss(out_conv_source.transpose(1, 2), out_conv_target.transpose(1, 2), labels_source[idx_source])

            if torch.isnan(loss).any() or torch.isinf(loss).any():
                ### Catch numerical errors/overflows (should not happen)
                if torch.isnan(loss).any():
                    logging.info("Nan loss")
                else:
                    logging.info("Inf loss")
                break

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if return_losses:
                losses.append(loss.item())

            if target_true is not None:
                maes_target[i] = MAE(target_filled, target_true, mask_target).item()
                rmses_target[i] = RMSE(target_filled, target_true, mask_target).item()
            
            if source_true is not None:
                maes_source[i] = MAE(source_filled, source_true, mask_source).item()
                rmses_source[i] = RMSE(source_filled, source_true, mask_source).item()

            if verbose and (i % report_interval == 0):
                if target_true is not None:
                    logging.info(f'Iteration {i}:\t Loss: {loss.item() / self.n_pairs:.4f}\t '
                                 f'Validation MAE: {maes_target[i]:.4f}\t'
                                 f'RMSE: {rmses_target[i]:.4f}')
                else:
                    logging.info(f'Iteration {i}:\t Loss: {loss.item() / self.n_pairs:.4f}')

        target_filled = target.detach().clone()
        target_filled[mask_target.bool()] = imps_target

        source_filled = source.detach().clone()
        source_filled[mask_source.bool()] = imps_source

        if target_true is not None:
            maes = maes_target
            rmses = rmses_target
        
        if return_losses:
            if target_true is not None and source_true is not None:
                return target_filled, source_filled, losses, maes, rmses
            else:
                return target_filled, source_filled, losses
        else:
            if target_true is not None and source_true is not None:
                return target_filled, source_filled, maes, rmses
            else:
                return target_filled, source_filled


class RR_MAD():
    """
    Round-Robin imputer with a batch MAD loss

    Parameters
    ----------
    models: iterable
        iterable of torch.nn.Module. The j-th model is used to predict the j-th
        variable using all others.
    
    num_class:

    CUDA_train:

    CUDA_MAD: 

    MAD_class:

    alpha:

    beta:

    torching:
        
    lr : float, default = 0.01
        Learning rate.

    opt: torch.nn.optim.Optimizer, default=torch.optim.Adam
        Optimizer class to use for fitting.
        
    max_iter : int, default=10
        Maximum number of round-robin cycles for imputation.

    niter : int, default=15
        Number of gradient updates for each model within a cycle.

    batchsize : int, defatul=128
        Size of the batches on which the sinkhorn divergence is evaluated.

    n_pairs : int, default=10
        Number of batch pairs used per gradient update.

    tol : float, default = 0.001
        Tolerance threshold for the stopping criterion.

    weight_decay : float, default = 1e-5
        L2 regularization magnitude.

    order : str, default="random"
        Order in which the variables are imputed.
        Valid values: {"random" or "increasing"}.


    """
    def __init__(self,
                 models_target,
                 models_source,
                 num_class,
                 MAD_class, 
                 lr=1e-2, 
                 opt=torch.optim.Adam, 
                 max_iter=10,
                 niter=15, 
                 batchsize=128,
                 n_pairs=10, 
                 tol=1e-3,
                 noise=0.1,
                 weight_decay=1e-5, 
                 order='random'):

        self.models_target = models_target
        self.models_source = models_source

        self.MAD_loss = MAD_loss(num_class,
                                 MAD_class,  
                                 target_prop=None,
                                 unbalanced=False, 
                                 un_reg=None, 
                                 un_mas=1.0)
        
        self.lr = lr
        self.opt = opt
        self.max_iter = max_iter
        self.niter = niter
        self.batchsize = batchsize
        self.n_pairs = n_pairs
        self.tol = tol
        self.noise = noise
        self.weight_decay=weight_decay
        self.order=order

        self.is_fitted = False

    def fit_transform(self, target, source, labels_source, 
                      verbose = True,
                      report_interval = 1, 
                      target_true = None, 
                      source_true = None,
                      return_losses = False ):
        """
            Imputes missing data in target and source using MAD_loss and fitted models.

        Parameters
        ----------
        target : torch.FloatTensor or torch.cuda.DoubleTensor, shape (n, t, d)
            Contains non-missing and missing data at the indices given by the
            "mask" argument. Missing values can be arbitrarily assigned 
            (e.g. with NaNs).

        source : torch.FloatTensor or torch.cuda.DoubleTensor, shape (n, t, d)
            Contains non-missing and missing data at the indices given by the
            "mask" argument. Missing values can be arbitrarily assigned 
            (e.g. with NaNs).
        
        labels_source :

        similarity_CE :

        verbose : bool, default=True
            If True, output loss to log during iterations.
            
        report_interval : int, default=1
            Interval between loss reports (if verbose).

        target_true: torch.FloatTensor or None, default=None
            Ground truth for the missing values. If provided, will output a 
            validation score during training. For debugging only.
        
        source_true: torch.FloatTensor or None, default=None
            Ground truth for the missing values. If provided, will output a 
            validation score during training. For debugging only.

        Returns
        -------
        target_filled: torch.FloatTensor or torch.cuda.FloatTensor
            Imputed missing data (plus unchanged non-missing data).

        source_filled: torch.FloatTensor or torch.cuda.FloatTensor
            Imputed missing data (plus unchanged non-missing data).

        """
        # target
        target = target.clone()
        n_target , t_target, d_target = target.shape
        td_target = t_target * d_target
        target = target.reshape(n_target, td_target)

        if target_true is not None:
            target_true = target_true.reshape(n_target, td_target)

        mask_target= torch.isnan(target).float()
        missing_pourcentage_target = mask_target.sum() / (n_target * td_target)
        
        # source
        source = source.clone()
        n_source, t_source, d_source = source.shape
        td_source = t_source * d_source
        source = source.reshape(n_source, td_source)

        if source_true is not None:
            source_true = source_true.reshape(n_source, td_source)

        mask_source = torch.isnan(source).float()
        missing_pourcentage_source = mask_source.sum() / (n_source * td_source)

        normalized_tol_target = self.tol * torch.max(torch.abs(target[~mask_target.bool()]))
        normalized_tol_source = self.tol * torch.max(torch.abs(source[~mask_source.bool()]))

        if return_losses:
            losses = []
        
        if self.batchsize > n_target // 2 or self.batchsize > n_source // 2:
            n = min(n_target, n_source)
            e = int(np.log2(n // 2))
            self.batchsize = 2**e
            if verbose:
                logging.info(f"Batchsize larger that half size = {max(n_target, n_source) // 2}. Setting batchsize to {self.batchsize}.")

        if verbose:
            logging.info(f"batchsize = {self.batchsize}")
            logging.info(f"Missing pourcentage target = {missing_pourcentage_target}")
            logging.info(f"Missing pourcentage source = {missing_pourcentage_source}")


        order_ = torch.argsort(mask_target.sum(0))

        # target
        optimizers_target = [self.opt(self.models_target[i].parameters(),
                             lr=self.lr, weight_decay=self.weight_decay) for i in range(td_target)]

        imps_target = (self.noise * torch.randn(mask_target.shape).float() + nanmean(target, 0))[mask_target.bool()]
        target[mask_target.bool()] = imps_target
        target_filled = target.clone()

        # source
        optimizers_source = [self.opt(self.models_source[i].parameters(),
                             lr=self.lr, weight_decay=self.weight_decay) for i in range(td_source)]

        imps_source = (self.noise * torch.randn(mask_source.shape).float() + nanmean(source, 0))[mask_source.bool()]
        source[mask_source.bool()] = imps_source
        source_filled = source.clone()

        if target_true is not None:
            maes_target = np.zeros(self.max_iter)
            rmses_target = np.zeros(self.max_iter)
        
        if source_true is not None:
            maes_source = np.zeros(self.max_iter)
            rmses_source = np.zeros(self.max_iter)

        data = {
                "source" : {
                            "n" : n_source,
                            "td" : td_source,
                            "mask" : mask_source,
                            "models" : self.models_source,
                            "optimizers" : optimizers_source,
                            "filled" : source_filled,
                            },
                "target" : {
                            "n" : n_target,
                            "td" : td_target,
                            "mask" : mask_target,
                            "models" : self.models_target,
                            "optimizers" : optimizers_target,
                            "filled" : target_filled
                            }
                }

        for i in range(self.max_iter):

            if self.order == 'random':
                order_ = np.random.choice(td_target + td_source, td_target + td_source, replace=False)
            target_old = data['target']['filled'].clone().detach()
            source_old = data['source']['filled'].clone().detach()

            loss = 0

            for l in range(td_target + td_source):
                j = order_[l].item()
                if j >= td_target:
                    working_data = "source"
                    j = j - td_target
                else:
                    working_data = "target"
                 
                n_not_miss = (~data[working_data]["mask"][:, j].bool()).sum().item()

                if data[working_data]["n"] - n_not_miss == 0:
                    continue  # no missing value on that coordinate

                for k in range(self.niter):

                    loss = 0

                    data[working_data]["filled"] = data[working_data]["filled"].detach()
                    data[working_data]["filled"][data[working_data]["mask"][:, j].bool(), j] = data[working_data]["models"][j](data[working_data]["filled"][data[working_data]["mask"][:, j].bool(), :][:, np.r_[0:j, j+1: data[working_data]["td"]]]).squeeze()

                    for _ in range(self.n_pairs):
                        
                        idx_target = np.random.choice(n_target, self.batchsize, replace=False)
                        out_conv_target = data["target"]["filled"].reshape(n_target, t_target, d_target)[idx_target]

                        idx_source = np.random.choice(n_source, self.batchsize, replace=False)
                        out_conv_source = data["source"]["filled"].reshape(n_source, t_source, d_source)[idx_source]

                        #print(working_data)
                        #print('out_conv_source.transpose(1, 2):\n\t',out_conv_source.transpose(1, 2))
                        #print('out_conv_target.transpose(1, 2):\n\t',out_conv_target.transpose(1, 2))


                        loss = loss + self.MAD_loss(out_conv_source.transpose(1, 2), out_conv_target.transpose(1, 2), labels_source[idx_source])

                    if torch.isnan(loss).any() or torch.isinf(loss).any():
                        ### Catch numerical errors/overflows (should not happen)
                        logging.info("Nan or inf loss")

                    data[working_data]["optimizers"][j].zero_grad()
                    loss.backward()
                    data[working_data]["optimizers"][j].step()

                    if return_losses:
                        losses.append(loss.item())

                # Impute with last parameters
                with torch.no_grad():
                    data[working_data]['filled'][data[working_data]['mask'][:, j].bool(), j] = data[working_data]['models'][j](data[working_data]['filled'][data[working_data]['mask'][:, j].bool(), :][:, np.r_[0:j, j+1: data[working_data]['td']]]).squeeze()

            if target_true is not None:
                maes_target[i] = MAE(data['target']['filled'], target_true, mask_target).item()
                rmses_target[i] = RMSE(data['target']['filled'], target_true, mask_target).item()
            
            if source_true is not None:
                maes_source[i] = MAE(data['source']['filled'], source_true, mask_source).item()
                rmses_source[i] = RMSE(data['source']['filled'], source_true, mask_source).item()

            if verbose and (i % report_interval == 0):
                if target_true is not None:
                    logging.info(f'Iteration {i}:\t Loss: {loss.item() / self.n_pairs:.4f}\t'
                                 f'Validation MAE: {maes_target[i]:.4f}\t'
                                 f'RMSE: {rmses_target[i]: .4f}')
                else:
                    logging.info(f'Iteration {i}:\t Loss: {loss.item() / self.n_pairs:.4f}')

            if (torch.norm(data['target']['filled'] - target_old, p=np.inf) < normalized_tol_target 
             or torch.norm(data['source']['filled'] - source_old, p=np.inf) < normalized_tol_source):
                break

        if i == (self.max_iter - 1) and verbose:
            logging.info('Early stopping criterion not reached')

        self.is_fitted = True

        if target_true is not None:
            maes = maes_target
            rmses = rmses_target
        
        if return_losses:
            if target_true is not None and source_true is not None:
                return data['target']['filled'], data['source']['filled'], losses, maes, rmses
            else:
                return data['target']['filled'], data['source']['filled'], losses
        else:
            if target_true is not None and source_true is not None:
                return data['target']['filled'], data['source']['filled'], maes, rmses
            else:
                return data['target']['filled'], data['source']['filled']


if __name__ == "__main__":
    import torch.nn as nn
    import os
    
    def from_numpy_to_torch(filename, float_or_long=True):
        data = np.load(filename)
        data_t = torch.from_numpy(data)
        if float_or_long:
            data_t = data_t.type(torch.float)
        else:
            data_t = data_t.type(torch.long)
        return data_t
    

    dirname = os.path.dirname(__file__)
    path = os.path.join(dirname, 'MAD_src/data/')
    name = 'remotes7cl'

    source = from_numpy_to_torch(path+name+'_11train.npy')
    source = source[:10000]
    labels = from_numpy_to_torch(path+name+'_11train_labels.npy')
    labels = labels[:10000]
    target = from_numpy_to_torch(path+name+'_14train.npy')
    target = target[:10000]
    
    np.random.seed(42)
    missing_pourcentage = 0.5

    n_target, t_target, d_target = target.shape

    # Creation of the mask for target
    mask_target = np.zeros((n_target, t_target, d_target), dtype = bool)
    missing_time = np.random.rand(t_target) < missing_pourcentage

    for t_, missing in enumerate(missing_time):
        if missing:
            mask_target[:, t_] = np.random.rand(d_target) < 0.90
            mask_target[:, t_][np.random.randint(0, d_target)] = False

    target_miss = np.copy(target)
    target_miss[mask_target] = np.nan
    Target_miss = torch.from_numpy(target_miss)

    n_source, t_source, d_source = source.shape
    mask_source = np.zeros((n_source, t_source, d_source), dtype = bool)
    missing_time_source = np.random.rand(t_source) < missing_pourcentage

    for t_, missing in enumerate(missing_time_source):
        if missing:
            mask_source[:, t_] = np.random.rand(d_source) < 0.90
            mask_source[:, t_][np.random.randint(0, d_source)] = False
        
    source_miss = np.copy(source)
    source_miss[mask_source] = np.nan
    Source_miss = torch.from_numpy(source_miss)

    #madot = OT_MAD(len(torch.unique(labels)), False, niter=2)
    #_, _, _, _ = madot.fit_transform(Target_miss, Source_miss, labels, target_true= target, source_true= source)

    #Create the imputation models
    td_target = t_target * d_target
    td_target_ = td_target - 1
    models_target = {}

    for i in range(td_target):
        models_target[i] = nn.Linear(td_target_, 1)

    td_source = t_source * d_source
    td_source_ = td_source - 1
    models_source = {}

    for i in range(td_source):
        models_source[i] = nn.Linear(td_source_, 1)
    # Imputation using MAD.

    #madrr = RR_MAD(models_target, models_source, len(torch.unique(labels)), False)
    #_, _, madrr_maes, madrr_rmses = madrr.fit_transform(Target_miss, Source_miss, labels, target_true = target, source_true = source)
