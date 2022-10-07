from test_runner import test_runner
from from_numpy_to_torch import from_numpy_to_torch

import torch

torch.set_default_tensor_type('torch.FloatTensor')

import os
import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.debug("test")

random_seed = 42

dirname = os.path.dirname(__file__)
resultat_path = os.path.join(dirname, 'resultat/')
path = os.path.join(dirname, 'MAD_src/data/')
name = 'remotes7cl'

missing_pourcentage_list = [0.5, 0.6, 0.7, 0.8, 0.9]
couple_liste= [('11','13'),('12','13'),('14','13'),('13','11'),('12','11')]

data = {'target' : None,
        'source' : None,
        'labels' : None}


for missing_pourcentage in missing_pourcentage_list:
    for couple in couple_liste:
        target_number, source_number = couple

        data['target'] =  from_numpy_to_torch(path + name + '_' + target_number + 'train.npy').detach().numpy()
        data['source'] = from_numpy_to_torch(path + name + '_' + source_number + 'train.npy').detach().numpy()
        data['labels'] = from_numpy_to_torch(path + name + '_' + source_number + 'train_labels.npy').detach().numpy()

        test_runner('RR MAD', data, missing_pourcentage, models_name='mlp', resultat_path = resultat_path, folder_name = target_number + '_' + source_number + f'_RR_MAD_mlp_{missing_pourcentage*100}', seed = random_seed)