import os
import torch
import h5py
import numpy as np
from torch.utils.data import Dataset


class DataLoaderTrain(Dataset):
    def __init__(self, base_dir, transform=None) -> None:
        super(DataLoaderTrain).__init__()

        self.transform = transform
        self.list_dir = os.path.join(base_dir, "list")
        self.sample_list = open(self.list_dir + '/train.txt').readlines()
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        slice_name = self.sample_list[idx].strip('\n')
        data_path = os.path.join(self.data_dir, 'train_npz', slice_name +'.npz')
        data = np.load(data_path)
        image, label = data['image'], data['label']

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample


class DataLoaderVal(Dataset):
    def __init__(self,base_dir) -> None:
        super(DataLoaderVal).__init__()
        self.list_dir = os.path.join(base_dir, "list")
        self.sample_list = open(self.list_dir + '/test_vol.txt').readlines()
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        vol_name = self.sample_list[idx].strip('\n')
        filepath = self.data_dir + '/test_vol_h5' + "/{}.npy.h5".format(vol_name)
        data = h5py.File(filepath)
        image, label = data['image'][:], data['label'][:]

        sample = {'image': image, 'label': label}
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample

