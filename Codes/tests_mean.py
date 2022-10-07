from test_runner import test_runner
from from_numpy_to_torch import from_numpy_to_torch

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
target_liste = ['11', '12', '13', '14']

data = {'target' : None,
        'source' : None,
        'labels' : None}
        

for missing_pourcentage in missing_pourcentage_list:
    for target_number in target_liste:

        data['target'] = from_numpy_to_torch(path + name + '_' + target_number + 'train.npy').detach().numpy()
            
        test_runner('Mean', data, missing_pourcentage, resultat_path = resultat_path, folder_name = target_number + f'_Mean_{missing_pourcentage*100}', seed = random_seed)