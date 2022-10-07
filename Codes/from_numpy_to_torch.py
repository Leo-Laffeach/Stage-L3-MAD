import torch
import numpy as np

def from_numpy_to_torch(filename, float_or_long=True):
        data = np.load(filename)
        data_t = torch.from_numpy(data)
        if float_or_long:
            data_t = data_t.type(torch.float)
        else:
            data_t = data_t.type(torch.long)
        return data_t