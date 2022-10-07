import numpy as np
import matplotlib.pyplot as plt

import os

data_set_color = {
                  '11': 'red',
                  '12': 'limegreen',
                  '13': 'blue',
                  '14': 'yellow',
                  '11_12': 'darkgreen',
                  '11_13': 'blue',
                  '13_11': 'red',
                  '13_12': 'limegreen',
                  '13_14': 'yellow'
                  }

data_set_list = ['11_12', '11_13', '13_11', '13_12', '13_14']
data_set_name = ['FR1 -> FR2', 'FR1 -> DK1', 'DK1 -> FR1', 'DK1 -> FR2', 'DK1 -> AT1']

data_set_target = {
                    'FR1' : ['DK1 -> FR1'],
                    'FR2' : ['FR1 -> FR2', 'DK1 -> FR2'],
                    'DK1' : ['FR1 -> DK1'],
                    'AT1' : ['DK1 -> AT1']
                  }

data_set_tgt_list = {
                    'FR1' : ['13_11'],
                    'FR2' : ['11_12', '13_12'],
                    'DK1' : ['11_13'],
                    'AT1' : ['13_14']
                    }

method_symbol = {
                 'OT_Muzellec': '.',
                 'OT_Muzellec_ar': '.',
                 'RR_Muzellec_linear': 'o',
                 'RR_Muzellec_mlp': '8',
                 'OT_MAD': '*',
                 'OT_MAD_ar': '*',
                 'RR_MAD_linear': 'x',
                 'RR_MAD_mlp': '+',
                 'OT_Muzellec_MAD' : '^',
                 'OT_Muzellec_MAD_ar' : '^',
                 'OT_MAD_early_stop': '*',
                 'OT_MAD_early_stop_ar': '*',
                 'OT_Muzellec_MAD_early_stop': '^',
                 'OT_Muzellec_MAD_early_stop_ar': '^'
                }

#method_list = ['Mean', 'OT_Muzellec', 'OT_MAD', 'OT_Muzellec_MAD', 'OT_MAD_early_stop', 'OT_Muzellec_MAD_early_stop']
#method_list = ['Mean_ar', 'OT_Muzellec_ar', 'OT_MAD_ar', 'OT_Muzellec_MAD_ar', 'OT_MAD_early_stop_ar', 'OT_Muzellec_MAD_early_stop_ar']
#method_list = ['Mean', 'OT_Muzellec', 'OT_MAD', 'OT_Muzellec_MAD', 'Mean_ar', 'OT_Muzellec_ar', 'OT_MAD_ar', 'OT_Muzellec_MAD_ar']
#method_list = ['Mean', 'OT_Muzellec', 'OT_MAD', 'OT_Muzellec_MAD']
method_list = ['Mean_ar', 'OT_Muzellec_ar', 'OT_MAD_ar', 'OT_Muzellec_MAD_ar']
#method_list = ['Mean', 'OT_Muzellec', 'OT_MAD_early_stop', 'OT_Muzellec_MAD_early_stop']
#method_list = ['Mean_ar', 'OT_Muzellec_ar', 'OT_MAD_early_stop_ar', 'OT_Muzellec_MAD_early_stop_ar']
ml = ['Moyenne', 'Sinkhorn', 'MAD', 'MAD + Sinkhorn']

data_correspondance = {
                        '11': 'FR1',
                        '12': 'FR2',
                        '13': 'DK1',
                        '14': 'AT1'
                      }

data_correspondance_reverse = {
                               'FR1': '11',
                               'FR2': '12',
                               'DK1': '13',
                               'AT1': '14'
                               }

def read_resultat(file):
    """
    Parameters
    ----------

    file: str
        The file path to read and to convert into a numpy array.
        format: 
        [
            value1
            value2
            ...
        ]

    Returns
    -------

    values: numpy.array
        An array with the data in file.
    """
    values = []
    if os.path.exists(file):
        with open(file, 'r') as data:
            data.readline() # remove the first '['.
            line = data.readline()
            while line != ']':
                values.append(float(line))
                line = data.readline()
    
    return np.array(values)




def show_resultat(path, target = 'AT1', file_type_name = 'maes', missing_pourcentage = 50., n_points = 2000, valide = False):
    """
    Parameters
    ----------

    path: str
        The folders path containing folders with maes.txt or rmses.txt files.
    
    file_type_name: str, default = 'maes'
        The name of the data file that we want to read. Either 'maes' or 'rmses'.
        Files must be in .txt.
    
    missing_pourcentage: float, default = 50.
        The pourcentage of missing data that we want to show.
    
    n_points: int, default = 2000
        Number of iteration for OT.
    
    """
    values_dico = {}

    if target is None:
        target = ['FR1', 'FR2', 'DK1', 'AT1']
    else:
        target = [target]

    for dirs in os.listdir(path):
        if not os.path.isdir(os.path.join(path, dirs)):
            continue

        name = (dirs.split('/')[-1]).split('_')
        if name[1] in data_correspondance:
            target_dirs = data_correspondance[name[1]]
        else:
            target_dirs = data_correspondance[name[0]]
        if target_dirs not in target:
            continue

        pourcentage_dirs = float(dirs.split('_')[-1])
        if missing_pourcentage != pourcentage_dirs:
            continue
        
        file_path = os.path.join(dirs, file_type_name)
        values_dico[f'{file_type_name}_{dirs}'] = read_resultat(os.path.join(path, file_path + '.txt'))
        if valide and os.path.exists(os.path.join(path,os.path.join(dirs, 'rmses_valide.txt'))):
            values_dico[f'{"rmses_valide"}_{dirs}'] = read_resultat(os.path.join(path,os.path.join(dirs, 'rmses_valide.txt')))

    fig, graph = plt.subplots()

    for value_name in values_dico:
        split_name = value_name.split('_')
        
        data_set = ''
        method = ''

        is_valide_data = split_name[1] == 'valide'
        k = 2 if is_valide_data else 1
        
        for info in split_name[k:-1]:
            if info in data_correspondance: # At least one and it's the first
                if data_set == '':
                    data_set += info
                else:
                    data_set += f'_{info}'

            else:
                if info != 'valide':
                    if method == '':
                        method += info
                    else:
                        method += f'_{info}'
            

        if method in method_list:
            n_points_value = len(values_dico[value_name])
            if n_points_value == 1:
                graph.hlines(xmin = 0, xmax = n_points,
                            y = values_dico[value_name][0], 
                            ls = '--', color=data_set_color[data_set])
            else:
                X = np.linspace(0, 1, n_points_value)*n_points_value
                space = 200
                N = len(X)//space
                Y = values_dico[value_name]
                if is_valide_data:
                    graph.plot(X,
                               Y,
                               color = data_set_color[data_set],
                               ls = ':')

                    min_index = np.argmin(Y)
                    graph.plot(X[min_index],
                               Y[min_index],
                               markersize = 8,  marker =  method_symbol[method], mec = 'black',
                               color = 'indigo')
                    
                else:
                    graph.plot(X, 
                               Y,
                               color = data_set_color[data_set])

                    graph.plot([X[i*space] for i in range(N)],
                               [Y[i*space] for i in range(N)],
                               markersize = 8,  marker =  method_symbol[method], mec = 'black',
                               ls = '', color = data_set_color[data_set])

    f = lambda m,c,ls: plt.plot([],[],marker=m, color=c, ls=ls)[0]
    liste_tgt = (np.array([data_set_tgt_list[j] for j in target])).flatten()
    handles = [f('', data_set_color[i], '-') for i in liste_tgt]
    handles += [f('', 'k', '--')]
    handles += [f(method_symbol[i], "k", '') for i in method_list[1:]]
    handles += [f('', 'k', ':')]
    labels = []
    for tgt in target:
        labels += data_set_target[tgt]
    labels += ml 
    if valide:
        labels += ['valide']
    graph.tick_params(axis='x', labelsize = 18)
    graph.set_xlabel('Itérations', fontsize = 20)
    graph.tick_params(axis='y', labelsize = 18)
    graph.legend(handles, labels, fontsize = 20, loc='center right', bbox_to_anchor=(1.135, .5))

    plt.show()
    

def show_resultat_value(path, n_points, file_type_name = 'rmses',  missing_pourcentage = 50.0):
    """
    Parameters
    ----------

    path: str
        The folders path containing folders with maes.txt or rmses.txt files.
    
    n_points: int
        The iteration value to show.
    
    file_type_name: str
        The name of the data file that we want to read. Either 'maes' or 'rmses'.
        Files must be in .txt.
        If None, show all the available file.
    
    missing_pourcentage: float, default = 50.0
        The pourcentage of missing data that we want to show.
    
    """
    values_dico = {}
    for dirs in os.listdir(path):
        if not os.path.isdir(os.path.join(path, dirs)):
            continue

        pourcentage_dirs = float(dirs.split('_')[-1])
        if missing_pourcentage is not None and missing_pourcentage != pourcentage_dirs:
            continue
        
        file_name_list = ['maes', 'rmses'] if file_type_name is None else [file_type_name]
        
        for file_name in file_name_list:
            file_path = os.path.join(dirs, file_name)
            values_dico[f'{file_name}_{dirs}'] = read_resultat(os.path.join(path, file_path + '.txt'))
    print(f'{file_type_name} with a missing pourcentage of {missing_pourcentage}%')
    for value_name in values_dico:
        split_name = value_name.split('_')
        value_file_name = split_name[0]

        label = ''
        method = ''
        for info in split_name[1:-1]:
            if info in data_correspondance: # At least one and it's the first
                if label == '':
                    label += data_correspondance[info]
                else:
                    label += f'->{data_correspondance[info]}'
            else:
                if method == '':
                    method += info
                else:
                    method += f'_{info}'
            
        if method in method_list:
            if n_points > len(values_dico[value_name]):
                print(f'{method}\t{label}:\n\t{values_dico[value_name][-1]}')
            else:
                print(f'{method}\t{label}:\n\t{values_dico[value_name][n_points]}')

        



if __name__ == '__main__':
    import os

    dirname = os.path.dirname(__file__)
    resultat_path = os.path.join(dirname, 'resultat/')

    file_type_name_list = ['maes', 'rmses']
    file_type_name = 'rmses'

    missing_pourcentage_list = [50, 60., 70., 80., 90.]
    missing_pourcentage = 50.

    target_list = ['FR1', 'FR2', 'DK1', 'AT1']
    target = 'FR1'

    n_iter = 10000

    show_graph = False
    show_value = False

    for missing_pourcentage in missing_pourcentage_list:
        for target in target_list:
            if show_graph:
                show_resultat(  resultat_path, 
                                target = target,
                                file_type_name = file_type_name, 
                                missing_pourcentage = missing_pourcentage   )

            if show_value:
                show_resultat_value(    resultat_path, 
                                        n_iter, 
                                        file_type_name = file_type_name,
                                        missing_pourcentage = missing_pourcentage   )
    


