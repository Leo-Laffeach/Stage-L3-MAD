import torch
import numpy as np


class DataLoader:
    def __init__(self, path):
        self.path = path
        self.datasets = {}

    def add_datasets(self, name, chan, n_classse):
        dataset = {"name": name,
                   "channel": chan,
                   "classes": n_classse,
                   "path": self.path + name + "_"}
        self.datasets[name] = dataset

    @staticmethod
    def from_numpy_to_torch(filename, float_or_long=True):
        data = np.load(filename)
        data_t = torch.from_numpy(data)
        if float_or_long:
            data_t = data_t.type(torch.float)
        else:
            data_t = data_t.type(torch.long)
        return data_t

    def load_dataset(self, name, src, tgt, sets):
        source = self.from_numpy_to_torch(self.datasets[name]["path"] + str(src) + sets + '.npy')
        source_label = self.from_numpy_to_torch(self.datasets[name]["path"] + str(src) + sets + '_labels.npy',
                                                float_or_long=False)

        target = self.from_numpy_to_torch(self.datasets[name]["path"] + str(tgt) + sets + '.npy')
        target_label = self.from_numpy_to_torch(self.datasets[name]["path"] + str(tgt) + sets + '_labels.npy',
                                                float_or_long=False)

        return source, source_label, target, target_label


if __name__ == "__main__":
    path = '/home/adr2.local/painblanc_f/codats-master/datas/numpy_data/'
    name = "tarnbzh5k"
    chan = 10
    classe = 5
    data = DataLoader(path)
    data.add_datasets(name, chan, classe)
    source, source_label, target, target_label = data.load_dataset(name, "11", "12", "train")
    print(source.shape, np.unique(source_label))
