import numpy as np

import torch
import logging

from MAD_loss_ import MAD_loss
from geomloss import SamplesLoss

from utils import nanmean, MAE, RMSE

class OT_Muzellec_MAD():
    """
        Calcul the loss with sinkhorn and MAD.
    Parameters
    ----------

    eps: float, default=0.01
        Sinkhorn regularization parameter.
        
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
                 eps=0.01,
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

        self.sk = SamplesLoss("sinkhorn", p=2, blur=eps, scaling=scaling, backend="tensorized")

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
        source : torch.DoubleTensor or torch.cuda.DoubleTensor, shape (n, t, d)
            Contains non-missing and missing data at the indices given by the
            "mask" argument. Missing values can be arbitrarily assigned 
            (e.g. with NaNs).
        
        target : torch.DoubleTensor or torch.cuda.DoubleTensor, shape (n, t, d)

        mask_target: torch.DoubleTensor or torch.cuda.DoubleTensor, shape (n, t, d)
            mask_target[i,j,k] == 1 if source[i,j,k] is missing, else mask_target[i,j,k] == 0.
        
        labels_source : 
            needed but not use.

        similarity_CE :

        verbose : bool, default=True
            If True, output loss to log during iterations.
            
        report_interval : int, default=1
            Interval between loss reports (if verbose).

        source_true: torch.DoubleTensor or None, default=None
            Ground truth for the missing values. If provided, will output a 
            validation score during training. For debugging only.

        Returns
        -------
        X_filled: torch.DoubleTensor or torch.cuda.DoubleTensor
            Imputed missing data (plus unchanged non-missing data).

        """
        # target
        target = target.clone()
        n_target , t_target, d_target = target.shape
        td_target = t_target * d_target
        target = target.reshape(n_target, td_target)
        target_true = target_true.reshape(n_target, td_target)

        mask_target= torch.isnan(target).float()
        missing_pourcentage_target = mask_target.sum() / (n_target * td_target)
        if verbose:
            logging.info(f"Pourcentage of missing value in the target: {missing_pourcentage_target}")

        # source
        source = source.clone()
        n_source, t_source, d_source = source.shape
        td_source = t_source * d_source 
        source = source.reshape(n_source, td_source)
        source_true = source_true.reshape(n_source, td_source)

        mask_source = torch.isnan(source).float()
        missing_pourcentage_source = mask_source.sum() / (n_source * td_source)
        if verbose:
            logging.info(f"Pourcentage of missing value in the source: {missing_pourcentage_source}")

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
                idx_target2 = np.random.choice(n_target, self.batchsize, replace=False)
                idx_source = np.random.choice(n_source, self.batchsize, replace=False)

                X1 = target_filled[idx_target]
                X2 = target_filled[idx_target2]
    
                out_conv_target = target_filled.reshape(n_target, t_target, d_target)[idx_target]
                out_conv_source = source_filled.reshape(n_source, t_source, d_source)[idx_source]
    
                loss = ( loss 
                       + self.MAD_loss(out_conv_source.transpose(1, 2), out_conv_target.transpose(1, 2), labels_source[idx_source])
                       + self.sk(X1, X2) / (0.5*(X1.shape[-1] + X2.shape[-1])))

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
