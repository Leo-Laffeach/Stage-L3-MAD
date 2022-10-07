import numpy as np
import torch.nn as nn

import torch
import os
import shutil

from sklearn.impute import SimpleImputer
from imputers import OTimputer, RRimputer
from MAD_imputers import OT_MAD, RR_MAD
from Muzellec_MAD_imputer import OT_Muzellec_MAD
from utils import *

imputers = [
            'Mean',
            'OT Muzellec',
            'RR Muzellec',
            'OT MAD',
            'RR MAD',
            'OT Muzellec MAD'
            ]

def test_runner(imputer_name, 
                data, 
                missing_pourcentage, 
                models_name = None, 
                batchsize = 128, 
                lr = 1e-2,
                resultat_path = '',
                folder_name = '',
                seed = None
                ):
    """
    Parameters
    ----------

    imputer_name : str,
        Choose between 'Mean', 'OT Muzelle', 'OT MAD', 'RR Muzellec', 'RR MAD' & 'OT Muzellec MAD'.
    
    data : dictionnary
        Contain 'target', 'source' & 'labels'
        with None or a pytorch array of dimension (n, t, d).
        No missing value.
    
    missing_pourcentage : float
        A number between 0 & 1
    
    models_name : str, default None
        Used if RR imputer. Either 'linear' or 'mlp'
    
    resultat_path : str, default ''
        A folder path for saving the resultat.
    
    folder_name : str, default ''
        The name of the folder where the resultat will be save.
    
    seed : int, default None
        Set the value of the random seed.

    """
    if seed is not None:
        np.random.seed(seed)
    
    # "data loaders"
    target = data['target']
    Target = torch.from_numpy(target)
    source = data['source']
    labels = data['labels']

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
    
    # create a mask for source too (different than target's one)
    if source is not None:
        Source = torch.from_numpy(source)
        Labels = torch.from_numpy(labels)

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

    # Imputer:
    if not(imputer_name in imputers):
        exit('Parametre Error! invalid imputer name.')

    imputer = None
    models = None
    DTW = None

    # Mean
    if imputer_name == 'Mean':
        target_miss = target_miss.reshape(n_target, t_target * d_target)
        imp = SimpleImputer().fit_transform(target_miss)
        target = target.reshape(n_target, t_target * d_target)
        mask_target = mask_target.reshape(n_target, t_target * d_target)

        maes = np.array([MAE(imp, target, mask_target)])
        rmses = np.array([RMSE(imp, target, mask_target)])

    # OT Muzellec
    if imputer_name == 'OT Muzellec':
        Target_miss = Target_miss.reshape(n_target, t_target*d_target).double()
        Target = Target.reshape(n_target, t_target*d_target).double()

        #epsilon = pick_epsilon(Target_miss)
        epsilon = 0.01
        
        imputer = OTimputer(eps=epsilon, batchsize=batchsize, lr=lr, niter=2000)
        imp, maes, rmses = imputer.fit_transform(Target_miss, verbose=True, report_interval=500, X_true=Target)

    # RR Muzellec
    if imputer_name == 'RR Muzellec':
        Target_miss = Target_miss.reshape(n_target, t_target*d_target).double()
        Target = Target.reshape(n_target, t_target*d_target).double()

        # Models generation:
        if models_name is None:
            print('Parametre Error: No model name')
            return None

        td_target = t_target*d_target
        td_target_ = td_target - 1
        models = {}

        if models_name == 'linear':
            for i in range(td_target):
                models[i] = nn.Linear(td_target_, 1)
        
        if models_name == 'mlp':
            for i in range(td_target):
                models[i] = nn.Sequential(nn.Linear(td_target_, 2 * td_target_),
                                        nn.ReLU(),
                                        nn.Linear(2 * td_target_, td_target_),
                                        nn.ReLU(),
                                        nn.Linear(td_target_, 1))

        if models == {}:
            print('Parametre Error: No model for RR Muzellec.')
            return None

        #epsilon = pick_epsilon(Target_miss)
        epsilon = 0.01

        imputer = RRimputer(models, eps=epsilon, lr=lr)
        imp, maes, rmses = imputer.fit_transform(Target_miss, verbose=True, X_true=Target)
        models = [imputer.models]

    # OT MAD
    if imputer_name == 'OT MAD':
        imputer = OT_MAD(len(torch.unique(Labels)), False, batchsize=batchsize, lr=lr, niter=2000)
        imp_target, imp_source, maes, rmses = imputer.fit_transform(Target_miss, Source_miss, Labels, target_true = Target, source_true = Source, report_interval = 500)
        DTW = imputer.MAD_loss.DTW_

    # RR MAD
    if imputer_name == 'RR MAD':
        # Models generation:
        if models_name is None:
            print('Parametre Error: No model name')
            return None

        td_target = t_target * d_target
        td_target_ = td_target - 1

        td_source = t_source * d_source
        td_source_ = td_source - 1

        models_target = {}
        models_source = {}

        if models_name == 'linear':
            for i in range(td_target):
                models_target[i] = nn.Linear(td_target_, 1)
            
            for i in range(td_source):
                models_source[i] = nn.Linear(td_source_, 1)

        
        if models_name == 'mlp':
            for i in range(td_target):
                models_target[i] = nn.Sequential(nn.Linear(td_target_, 2 * td_target_),
                                                 nn.ReLU(),
                                                 nn.Linear(2 * td_target_, td_target_),
                                                 nn.ReLU(),
                                                 nn.Linear(td_target_, 1))
            
            for i in range(td_source):
                models_source[i] = nn.Sequential(nn.Linear(td_source_, 2 * td_source_),
                                                 nn.ReLU(),
                                                 nn.Linear(2 * td_source_, td_source_),
                                                 nn.ReLU(),
                                                 nn.Linear(td_source_, 1))

        if models_target == {} or models_source == {}:
            print('Parametre Error: No model for RR MAD.')
            return None
               
        Labels = torch.from_numpy(labels)
        imputer = RR_MAD(models_target, models_source, len(torch.unique(Labels)), False, lr=lr)
        imp_target, imp_source, maes, rmses = imputer.fit_transform(Target_miss, Source_miss, Labels, target_true = Target, source_true = Source, report_interval = 500)
        models = [imputer.models_target, imputer.models_source]
        DTW = imputer.MAD_loss.DTW_

    # OT Muzellec MAD
    if imputer_name == 'OT Muzellec MAD':
        epsilon = 0.01

        imputer = OT_Muzellec_MAD(len(torch.unique(Labels)), False, batchsize=batchsize, eps=epsilon, lr=lr, niter=2000)
        imp_target, imp_source, maes, rmses = imputer.fit_transform(Target_miss, Source_miss, Labels, target_true = Target, source_true = Source, report_interval = 500)

    # verify the existence of resultat_path, if not error
    if not os.path.exists(resultat_path):
        print('Error path name: No such directory')
        return None
    # if a folder with folder_name exist, delete it
    path_folder = resultat_path + folder_name
    if os.path.exists(path_folder):
        shutil.rmtree(path_folder)
    # create the folder with folder_name as name
    os.mkdir(path_folder)

    if models is not None:
        if len(models) == 1:
            os.mkdir(path_folder + "/models")
            for i in models[0]:
                torch.save(models[0][i].state_dict(), path_folder + f'/models/models_{i}.pt')
        if len(models) == 2:
            os.mkdir(path_folder + "/models_target")
            for i in models[0]:
                torch.save(models[0][i], path_folder + f'/models_target/models_target_{i}.pt')
            
            os.mkdir(path_folder + "/models_source")
            for i in models[1]:
                torch.save(models[1], path_folder + f'/models_source/models_source_{i}.pt')

    if DTW is not None:
        os.mkdir(path_folder + "/DTW")
        for i, dtw in enumerate(DTW):
            torch.save(dtw, path_folder + f'/DTW/previous_DTW_{i}')
    
    # save maes & rmses
    MAES = open(path_folder + "/maes.txt", "w+")
    RMSES = open(path_folder + "/rmses.txt", "w+")

    content_maes = '[\n' 
    for mae in maes:
        content_maes += '\t'
        content_maes += str(mae.item())
        content_maes += '\n'
    content_maes += ']'
    
    content_rmses = '[\n' 
    for rmse in rmses:
        content_rmses += '\t'
        content_rmses += str(rmse.item())
        content_rmses += '\n'
    content_rmses += ']'

    MAES.write(content_maes)
    RMSES.write(content_rmses)
    
    MAES.close()
    RMSES.close()

    # save parameter
    parameter = open(path_folder + "/parameter.txt", "w+")
    content = f'missing pourcentage = {missing_pourcentage}\nrandom seed = {seed}'
    parameter.write(content)
    parameter.close()

    return None

#▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

if __name__ == "__main__":

    from from_numpy_to_torch import from_numpy_to_torch

    random_seed = 42

    dirname = os.path.dirname(__file__)
    resultat_path = os.path.join(dirname, 'resultat/')
    path = os.path.join(dirname, 'MAD_src/data/')

    name = 'remotes7cl'
    target_number = '11'
    source_number = '12'

    N = 10000

    data = {
            'target' : from_numpy_to_torch(path + name + '_' + target_number + 'train.npy')[:N].detach().numpy(),
            'source' : from_numpy_to_torch(path + name + '_' + source_number + 'train.npy')[:N].detach().numpy(),
            'labels' : from_numpy_to_torch(path + name + '_' + source_number + 'train_labels.npy')[:N].detach().numpy()
            }

    missing_pourcentage = 0.5

    # Mean
    """
    print('Mean start:')
    test_runner('Mean', data, missing_pourcentage, resultat_path = resultat_path, folder_name = target_number + f'_Mean_{missing_pourcentage*100}', seed = random_seed)
    print('Mean end.')
    """
    # Muzellec
    """
    print('OT Muzellec start:')
    test_runner('OT Muzellec', data, missing_pourcentage, resultat_path = resultat_path, folder_name = target_number + f'_OT_Muzellec_{missing_pourcentage*100}', seed = random_seed)
    print('OT Muzellec end.')
    """
    #torch.set_default_tensor_type('torch.DoubleTensor')
    """
    print('RR Muzellec linear start:')
    test_runner('RR Muzellec', data, missing_pourcentage, models_name='linear', resultat_path = resultat_path, folder_name = target_number + f'_RR_Muzellec_linear_{missing_pourcentage*100}', seed = random_seed)
    print('RR Muzellec linear end.')
    """
    """
    print('RR Muzellec mlp start:')
    test_runner('RR Muzellec', data, missing_pourcentage, models_name='mlp', resultat_path = resultat_path, folder_name = target_number + f'_RR_Muzellec_mlp_{missing_pourcentage*100}', seed = random_seed)
    print('RR Muzellec mlp end:')
    """
    # MAD
    """
    print('OT MAD start:')
    test_runner('OT MAD', data, missing_pourcentage, resultat_path = resultat_path, folder_name = target_number + '_' + source_number + f'_OT_MAD_{missing_pourcentage*100}', seed = random_seed)
    print('OT MAD end.')
    """
    #torch.set_default_tensor_type('torch.FloatTensor')
    """
    print('RR MAD linear start:')
    test_runner('RR MAD', data, missing_pourcentage, models_name='linear', resultat_path = resultat_path, folder_name = target_number + '_' + source_number + f'_RR_MAD_linear_{missing_pourcentage*100}', seed = random_seed)
    print('RR MAD linear end.')
    """
    """
    print('RR MAD mlp start:')
    test_runner('RR MAD', data, missing_pourcentage, models_name='mlp', resultat_path = resultat_path, folder_name = target_number + '_' + source_number + f'_RR_MAD_mlp_{missing_pourcentage*100}', seed = random_seed)
    print('RR MAD mlp end.')
    """
    
   