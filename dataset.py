import os
import random
import math

import numpy as np
import torch
import torch.utils.data as data


# multi-modal same-class pairing strategy
def MSCP(he_data_list, ihc_data_list, num_classes):

    D_mm = []
    M = len(he_data_list) # or len(ihc_data_list)
    D_y = []

    for i in range(num_classes):
        sub_y_list = [i]*math.floor(M/num_classes)
        D_y = D_y + sub_y_list

    he_data_set = [[] for _ in range(num_classes)]
    ihc_data_set = [[] for _ in range(num_classes)]
    for he_id,he_y in he_data_list:
        he_data_set[he_y].append(he_id)
    for ihc_id,ihc_y in ihc_data_list:
        ihc_data_set[ihc_y].append(ihc_id)
    for i in range(math.floor(M/num_classes)*num_classes):
        y = D_y[i]
        random.shuffle(he_data_set[y])
        random.shuffle(ihc_data_set[y])
        he_id, ihc_id = he_data_set[y][0],ihc_data_set[y][0]
        D_mm.append((he_id,ihc_id,y))
    return D_mm




class TrainDataset(data.Dataset):
    def __init__(self, he_feature_path, ihc_feature_path, he_csv, ihc_csv, mscp, fold_k, sample_num=None, num_classes=4):
        super(TrainDataset, self).__init__()
        self.he_feature_path = he_feature_path
        self.ihc_feature_path = ihc_feature_path
        self.mscp = mscp
        self.fold_k = fold_k
        self.sample_num = sample_num
        self.num_classes = num_classes
        self.he_data_list = []
        self.ihc_data_list = []
        with open(he_csv, 'r') as f:
            for i in f.readlines():
                slide_id = i.split(',')[0]
                slide_label = int(i.split(',')[1].strip())
                fold = int(i.split(',')[-1].strip())
                if fold != self.fold_k:
                    self.he_data_list.append((slide_id, slide_id, slide_label))
        with open(ihc_csv, 'r') as f:
            for i in f.readlines():
                slide_id = i.split(',')[0]
                slide_label = int(i.split(',')[1].strip())
                fold = int(i.split(',')[-1].strip())
                if fold != self.fold_k:
                    self.ihc_data_list.append((slide_id, slide_id, slide_label))
        if not mscp:
            self.data_list = self.he_data_list # or self.ihc_data_list
        else:
            self.data_list = MSCP(self.he_data_list,self.ihc_data_list, self.num_classes)

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
    def __init__(self, he_feature_path, ihc_feature_path, he_csv, ihc_csv, val_mode, fold_k, mscp, num_classes=4):
        super(ValDataset, self).__init__()
        self.he_feature_path = he_feature_path
        self.ihc_feature_path = ihc_feature_path
        self.val_mode = val_mode
        self.fold_k = fold_k
        self.mscp = mscp
        self.num_classes = num_classes
        self.he_data_list = []
        self.ihc_data_list = []
        with open(he_csv, 'r') as f:
            for i in f.readlines():
                slide_id = i.split(',')[0]
                slide_label = int(i.split(',')[1].strip())
                fold = int(i.split(',')[-1].strip())
                if fold == self.fold_k:
                    self.he_data_list.append((slide_id, slide_id, slide_label))
        with open(ihc_csv, 'r') as f:
            for i in f.readlines():
                slide_id = i.split(',')[0]
                slide_label = int(i.split(',')[1].strip())
                fold = int(i.split(',')[-1].strip())
                if fold == self.fold_k:
                    self.ihc_data_list.append((slide_id, slide_id, slide_label))
        if self.val_mode == 'he':
            self.data_list = self.he_data_list
        elif self.val_mode == 'ihc':
            self.data_list = self.ihc_data_list
        elif self.val_mode == 'mm' and self.mscp: # multi-modal
            self.data_list = MSCP(self.he_data_list, self.ihc_data_list, self.num_classes)
        elif self.val_mode == 'mm' and not self.mscp: # multi-modal
            self.data_list = self.he_data_list # or self.ihc_data_list
        else:
            raise Exception("Invalid value for val mode!")


    def __getitem__(self, index: int):
        if self.val_mode == 'he':
            he_feature_data = torch.load(os.path.join(self.he_feature_path, self.data_list[index][0] + "_features.pth"))
            ihc_feature_data = torch.tensor(0)
            label = torch.LongTensor([self.data_list[index][2]])
        elif self.val_mode == 'ihc':
            he_feature_data = torch.tensor(0)
            ihc_feature_data = torch.load(os.path.join(self.ihc_feature_path, self.data_list[index][1] + "_features.pth"))
            label = torch.LongTensor([self.data_list[index][2]])
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
    def __init__(self, he_feature_path, ihc_feature_path, he_csv, ihc_csv, test_mode, mscp, num_classes=4):
        super(TestDataset, self).__init__()
        self.he_feature_path = he_feature_path
        self.ihc_feature_path = ihc_feature_path
        self.test_mode = test_mode
        self.mscp = mscp
        self.num_classes = num_classes
        self.he_data_list = []
        self.ihc_data_list = []
        with open(he_csv, 'r') as f:
            for i in f.readlines():
                slide_id = i.split(',')[0]
                slide_label = int(i.split(',')[-1].strip())
                self.he_data_list.append((slide_id, slide_id, slide_label))
        with open(ihc_csv, 'r') as f:
            for i in f.readlines():
                slide_id = i.split(',')[0]
                slide_label = int(i.split(',')[-1].strip())
                self.ihc_data_list.append((slide_id, slide_id, slide_label))
        if self.test_mode == 'he':
            self.data_list = self.he_data_list
        elif self.test_mode == 'ihc':
            self.data_list = self.ihc_data_list
        elif self.test_mode == 'mm' and not mscp: # multi-modal
            self.data_list = self.he_data_list # or self.ihc_data_list
        else:
            raise Exception("Invalid value for test mode!")

    def __getitem__(self, index: int):
        if self.test_mode == 'he':
            he_feature_data = torch.load(os.path.join(self.he_feature_path, self.data_list[index][0] + "_features.pth"))
            ihc_feature_data = torch.tensor(0)
            label = torch.LongTensor([self.data_list[index][2]])
        elif self.test_mode == 'ihc':
            he_feature_data = torch.tensor(0)
            ihc_feature_data = torch.load(
                os.path.join(self.ihc_feature_path, self.data_list[index][1] + "_features.pth"))
            label = torch.LongTensor([self.data_list[index][2]])
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