import numpy as np
import torch
import MADCNN
import dataloader
import os
import torch.nn as nn


class EarlyStopping:
    def __init__(self, data_name, data_path, chan, classe, source, target, sets, path_models, name_model, path_saving,
                 iteration_step, iteration_max, alpha_range, beta_range, lr_range):
        # First we create the datasets
        self.data_name = data_name
        self.data = dataloader.DataLoader(data_path)
        self.data.add_datasets(data_name, chan, classe)
        self.source, self.source_label, self.target, self.target_label = self.data.load_dataset(data_name,
                                                                                           source, target, sets)
        self.feature_extractor = nn.Sequential(
                nn.Conv1d(in_channels=self.data.datasets[data_name]["channel"], out_channels=128, kernel_size=8, stride=1,
                          padding="same", bias=False),
                nn.BatchNorm1d(num_features=128, affine=False),
                nn.ReLU(),

                nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, stride=1, padding="same", bias=False),
                nn.BatchNorm1d(num_features=256, affine=False),
                nn.ReLU(),

                nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding="same", bias=False),
                nn.BatchNorm1d(num_features=128, affine=False),
                nn.ReLU()
            )

        self.classifier = nn.Sequential(nn.Linear(128, self.data.datasets[data_name]["classes"]))

        self.path_models = path_models
        self.name_model = name_model
        self.path_saving = path_saving
        self.iteration_step = iteration_step
        self.iteration_max = iteration_max
        self.alpha_range = alpha_range
        self.beta_range = beta_range
        self.lr_range = lr_range

    def generate_pseudo_labels_target(self, path_s2s, path_s2s_comp, lr):
        # Get the pseudo label from the source2source model
        model = MADCNN.CNNMAD(name=path_s2s, batchsize=256,
                              feature_extractor=self.feature_extractor, classifier=self.classifier,
                              n_classes=self.data.datasets[self.data_name]["classes"], alpha=0.0, beta=0.0, lamb=1.,
                              MAD_class=True, supervised_validation=True, lr=lr, saving=False, CUDA_train=False)
        model.load_state_dict(torch.load(path_s2s + str(lr) + path_s2s_comp))
        pred_target = model.proxy_labels(self.target)
        return pred_target

    def evaluate_from_pseudo_label(self, alpha, beta, lr, iteration, proxy_target, n):
        # Evaluate the accuracy based on the proxies of target labels
        path_l = self.path_models + str(alpha) + "/" + str(beta) + "/" + str(lr) + "/" + self.name_model[n]
        path_s = self.path_saving + str(alpha) + "/" + str(beta) + "/" + str(lr) + "/" + self.name_model[n]
        model = MADCNN.CNNMAD(name=path_s, batchsize=256,
                              feature_extractor=self.feature_extractor, classifier=self.classifier,
                              n_classes=self.data.datasets[self.data_name]["classes"], alpha=alpha, beta=beta, lamb=1.,
                              MAD_class=True, supervised_validation=True, lr=lr, saving=True, CUDA_train=False)
        if os.path.exists(path_l + str(iteration) + ".pt"):
            model.load_state_dict(torch.load(path_l + str(iteration) + ".pt"))
        elif os.path.exists(path_l + str(iteration) + "best.pt"):
            model.load_state_dict(torch.load(path_l + str(iteration) + "best.pt"))
        else:
            return 0
            #  raise Exception("Model not found!")
        pred_target = model.proxy_labels(self.target)
        correct = pred_target.eq(proxy_target.data.view_as(pred_target)).cpu().sum()
        return 100. * correct / len(proxy_target)

    def general_evaluation(self, path_s2s, path_s2s_comp, path_s2s_comp2):
        #  dimension of the table is #alpha * #beta * #lambda * #(rep - 1)
        rep = int(self.iteration_max / self.iteration_step)
        perf_proxy = np.zeros(shape=(len(self.alpha_range), len(self.beta_range), len(self.lr_range), rep))
        ref_param = []
        accuracy_max = 0
        for lr in range(0, len(self.lr_range)):
            proxy_target = self.generate_pseudo_labels_target(path_s2s, path_s2s_comp, self.lr_range[lr])
            proxy_target2 = self.generate_pseudo_labels_target(path_s2s, path_s2s_comp2, self.lr_range[lr])
            for a in range(0, len(self.alpha_range)):
                for b in range(0, len(self.beta_range)):
                    for i in range(1, rep + 1):
                        accuracy = 0
                        it = i * self.iteration_step
                        for n in range(0, len(self.name_model)):
                            if n == 0:
                                accuracy += self.evaluate_from_pseudo_label(self.alpha_range[a], self.beta_range[b],
                                                                            self.lr_range[lr], it, proxy_target, n)
                            if n == 1:
                                accuracy += self.evaluate_from_pseudo_label(self.alpha_range[a], self.beta_range[b],
                                                                            self.lr_range[lr], it, proxy_target2, n)
                        accuracy /= len(self.name_model)
                        perf_proxy[a, b, lr, i - 1] = accuracy
                        name_ref_param = "alpha: " + str(self.alpha_range[a]) + " beta: " + str(self.beta_range[b]) + \
                                         " lr: " + str(self.lr_range[lr]) + " iteration: " + str(it)
                        ref_param.append(name_ref_param)
                        if accuracy > accuracy_max:
                            accuracy_max = accuracy
                            print(name_ref_param, accuracy_max)
        #  print(perf_proxy)
        best_model_pos = np.argmax(perf_proxy)
        #  print(best_model_pos)
        #  print(ref_param[best_model_pos])
        """best_alpha = self.alpha_range[best_model_pos[0]]
        best_beta = self.beta_range[best_model_pos[1]]
        best_lr = self.lr_range[best_model_pos[2]]
        best_it = self.iteration_step * (best_model_pos[-1] + 1)
        return best_alpha, best_beta, best_lr, best_it"""


if __name__ == "__main__":
    data_name = "tarnbzh5k"
    data_path = "/share/home/fpainblanc/MAD-CNN/numpy_data/"
    chan = 10
    classe = 5
    source = "11"
    target = "12"
    sets = "valid"
    path_models = "/share/home/fpainblanc/CNNMAD/TBZH5K/Torch/"
    name_model = ["batch_eq256", "batch_eq256_rep2"]
    path_saving = "/share/home/fpainblanc/CNNMAD/TBZH5K_acc/Torch/"
    iteration_step = 2000
    iteration_max = 10000
    alpha_range = [0.0, 0.1, 0.01, 0.001, 0.0001, 0.00001]
    beta_range = [0.0, 0.1, 0.01, 0.001, 0.0001]
    lr_range = [0.0001, 0.001]
    path_pre = "/share/home/fpainblanc/CNNMAD/TBZH5K/baseline/"
    path_pre_comp = "/batch_eq256_s2s30000.pt"
    path_pre_comp2 = "/batch_eq256_rep2_s2s30000.pt"
    tbzh_stop = EarlyStopping(data_name=data_name, data_path=data_path, chan=chan, classe=classe, source=source,
                              target=target, sets=sets, path_models=path_models, name_model=name_model,
                              path_saving=path_saving, iteration_step=iteration_step, iteration_max=iteration_max,
                              alpha_range=alpha_range, beta_range=beta_range, lr_range=lr_range)
    tbzh_stop.general_evaluation(path_pre, path_pre_comp, path_pre_comp2)

"""
if args.dataset == 'HAR':
    chan = 9
    n_classes = 6
if args.dataset == "TARNBZH":
        n_classes = 5
    chan = 10
if args.dataset == "TARNBZH5K":
        n_classes = 5
    chan = 10
if args.dataset == "TRACE":
    n_classes = 4
    chan = 1
if args.dataset == "REMOTES":
    n_classes = 8
    chan = 10
if args.dataset == "REMOTES7CL":
    n_classes = 7
    chan = 10
"""