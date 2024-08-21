import os
import random

import numpy as np
import torch
import torch.utils.data as data


# multi-modal same-class pairing strategy
def MSCP(he_data_list, ihc_data_list):
    return []




class TrainDataset(data.Dataset):
    def __init__(self, he_feature_path, ihc_feature_path, he_csv, ihc_csv, mscp, fold_k, sample_num=None):
        super(TrainDataset, self).__init__()
        self.he_feature_path = he_feature_path
        self.ihc_feature_path = ihc_feature_path
        self.mscp = mscp
        self.fold_k = fold_k
        self.sample_num = sample_num
        self.he_data_list = []
        self.ihc_data_list = []
        with open(he_csv, 'r') as f:
            for i in f.readlines():
                slide_feature_path = i.split(',')[0]
                slide_label = int(i.split(',')[1].strip())
                fold = int(i.split(',')[-1].strip())
                if fold != self.fold_k:
                    self.he_data_list.append((slide_feature_path, slide_label))
        with open(ihc_csv, 'r') as f:
            for i in f.readlines():
                slide_feature_path = i.split(',')[0]
                slide_label = int(i.split(',')[1].strip())
                fold = int(i.split(',')[-1].strip())
                if fold != self.fold_k:
                    self.ihc_data_list.append((slide_feature_path, slide_label))
        if not mscp:
            self.data_list = self.he_data_list # or self.ihc_data_list
        else:
            self.data_list = MSCP(self.he_data_list,self.ihc_data_list)

    def __getitem__(self, index: int):
        he_feature_data = torch.load(os.path.join(self.he_feature_path, self.data_list[index][0] + "_features.pth"))
        ihc_feature_data = torch.load(os.path.join(self.ihc_feature_path, self.data_list[index][1] + "_features.pth"))
        label = torch.LongTensor([self.data_list[index][2]])

        he_n, ihc_n = he_feature_data.shape[0], ihc_feature_data.shape[0]
        if self.sample_num is not None:
            if he_n > self.sample_num:
                he_select_index = torch.LongTensor(random.sample(range(he_n), self.sample_num))
            else:
                he_select_index = torch.LongTensor(random.sample(range(he_n), he_n))
            if ihc_n > self.sample_num:
                ihc_select_index = torch.LongTensor(random.sample(range(ihc_n), self.sample_num))
            else:
                ihc_select_index = torch.LongTensor(random.sample(range(ihc_n), ihc_n))
            he_feature_data = torch.index_select(he_feature_data, 0, he_select_index)
            ihc_feature_data = torch.index_select(ihc_feature_data, 0, ihc_select_index)
        return he_feature_data, ihc_feature_data, label

    def __len__(self):
        return len(self.data_list)

class ValDataset(data.Dataset):
    def __init__(self, he_feature_path, ihc_feature_path, he_csv, ihc_csv, val_mode, fold_k):
        super(ValDataset, self).__init__()
        self.he_feature_path = he_feature_path
        self.ihc_feature_path = ihc_feature_path
        self.val_mode = val_mode
        self.fold_k = fold_k
        self.he_data_list = []
        self.ihc_data_list = []
        with open(he_csv, 'r') as f:
            for i in f.readlines():
                slide_feature_path = i.split(',')[0]
                slide_label = int(i.split(',')[1].strip())
                fold = int(i.split(',')[-1].strip())
                if fold == self.fold_k:
                    self.he_data_list.append((slide_feature_path, slide_label))
        with open(ihc_csv, 'r') as f:
            for i in f.readlines():
                slide_feature_path = i.split(',')[0]
                slide_label = int(i.split(',')[1].strip())
                fold = int(i.split(',')[-1].strip())
                if fold == self.fold_k:
                    self.ihc_data_list.append((slide_feature_path, slide_label))
        if self.val_mode == 'he':
            self.data_list = self.he_data_list # or self.ihc_data_list
        elif self.val_mode == 'ihc':
            self.data_list = self.ihc_data_list  # or self.ihc_data_list
        elif self.val_mode == 'mm': # multi-modal
            self.data_list = MSCP(self.he_data_list,self.ihc_data_list)
        else:
            raise Exception("Invalid value for val mode!")


    def __getitem__(self, index: int):
        if self.val_mode == 'he':
            he_feature_data = torch.load(os.path.join(self.he_feature_path, self.data_list[index][0] + "_features.pth"))
            ihc_feature_data = None
            label = torch.LongTensor([self.data_list[index][1]])
        elif self.val_mode == 'ihc':
            he_feature_data = None
            ihc_feature_data = torch.load(os.path.join(self.ihc_feature_path, self.data_list[index][1] + "_features.pth"))
            label = torch.LongTensor([self.data_list[index][1]])
        elif self.val_mode == 'mm':
            he_feature_data = torch.load(os.path.join(self.he_feature_path, self.data_list[index][0] + "_features.pth"))
            ihc_feature_data = torch.load(os.path.join(self.ihc_feature_path, self.data_list[index][1] + "_features.pth"))
            label = torch.LongTensor([self.data_list[index][2]])
        else:
            raise Exception("Invalid value for val mode!")
        return he_feature_data, ihc_feature_data, label

    def __len__(self):
        return len(self.data_list)

class TestDataset(data.Dataset):
    def __init__(self, he_feature_path, ihc_feature_path, he_csv, ihc_csv, test_mode):
        super(TestDataset, self).__init__()
        self.he_feature_path = he_feature_path
        self.ihc_feature_path = ihc_feature_path
        self.test_mode = test_mode
        self.he_data_list = []
        self.ihc_data_list = []
        with open(he_csv, 'r') as f:
            for i in f.readlines():
                slide_feature_path = i.split(',')[0]
                slide_label = int(i.split(',')[-1].strip())
                self.he_data_list.append((slide_feature_path, slide_label))
        with open(ihc_csv, 'r') as f:
            for i in f.readlines():
                slide_feature_path = i.split(',')[0]
                slide_label = int(i.split(',')[-1].strip())
                self.ihc_data_list.append((slide_feature_path, slide_label))
        if self.test_mode == 'he':
            self.data_list = self.he_data_list # or self.ihc_data_list
        elif self.test_mode == 'ihc':
            self.data_list = self.ihc_data_list  # or self.ihc_data_list
        elif self.test_mode == 'mm': # multi-modal
            self.data_list = MSCP(self.he_data_list,self.ihc_data_list)
        else:
            raise Exception("Invalid value for test mode!")

    def __getitem__(self, index: int):
        if self.test_mode == 'he':
            he_feature_data = torch.load(os.path.join(self.he_feature_path, self.data_list[index][0] + "_features.pth"))
            ihc_feature_data = None
            label = torch.LongTensor([self.data_list[index][1]])
        elif self.test_mode == 'ihc':
            he_feature_data = None
            ihc_feature_data = torch.load(
                os.path.join(self.ihc_feature_path, self.data_list[index][1] + "_features.pth"))
            label = torch.LongTensor([self.data_list[index][1]])
        elif self.test_mode == 'mm':
            he_feature_data = torch.load(os.path.join(self.he_feature_path, self.data_list[index][0] + "_features.pth"))
            ihc_feature_data = torch.load(
                os.path.join(self.ihc_feature_path, self.data_list[index][1] + "_features.pth"))
            label = torch.LongTensor([self.data_list[index][2]])
        else:
            raise Exception("Invalid value for test mode!")
        return he_feature_data, ihc_feature_data, label

    def __len__(self):
        return len(self.data_list)