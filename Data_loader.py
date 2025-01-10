import numpy as np
import torch
import os
import scipy.io as sio
from sklearn import preprocessing
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader, random_split

class WDataset(Dataset):
    def __init__(self, ids, dir_input, offsets, fov=0.2, extension='.mat'):
        self.dir_input = dir_input
        self.extension = extension
        self.ids = ids
        self.offsets = offsets
        self.data_len = len(self.ids)
        self.fov = fov

    def __getitem__(self, index):
        id_input = self.dir_input + self.ids[index]
        input_mat = sio.loadmat(id_input + self.extension)
        wave = input_mat['wave']
        mu = input_mat['mu']
        mu = np.float64(mu)
        mfre = input_mat['mfre']
        mfre = np.float64(mfre)
        index = np.int64(input_mat['index'])
        # mask = input_mat['mask']
        fov = self.fov
        wave_number = fov * np.divide(mfre, ((mu/1000)**(1/2)), out=np.zeros_like(mu), where=mu!=0)
        input = torch.from_numpy(wave).float().unsqueeze(0)
        gt = torch.from_numpy(wave_number).float().unsqueeze(0)

        return input, gt, mfre, fov, index

    def __len__(self):
        return self.data_len

def get_dataloader_for_train(dir_input, offsets=8, fov=0.2, batch_size=10):
    ids = [f[:-4] for f in os.listdir(dir_input)]
    dset = WDataset(ids, dir_input, offsets, fov)
    dataloaders = {}
    dataloaders['train'] = DataLoader(dset, batch_size=batch_size, shuffle=True, drop_last = False)

    return dataloaders['train']

def get_dataloader_for_val(dir_input, offsets=8, fov=0.2, batch_size=10):
    ids = [f[:-4] for f in os.listdir(dir_input)]
    dset = WDataset(ids, dir_input, offsets, fov)
    dataloaders = {}
    dataloaders['val'] = DataLoader(dset, batch_size=batch_size, shuffle=True, drop_last = False)

    return dataloaders['val']

def get_dataloader_for_test(dir_input, offsets=8, fov=0.2, batch_size=10):
    ids = [f[:-4] for f in os.listdir(dir_input)]
    dset = WDataset(ids, dir_input, offsets, fov)
    dataloaders = {}
    dataloaders['test'] = DataLoader(dset, batch_size=batch_size, shuffle=True, drop_last = False)

    return dataloaders['test']
