import sys
import datetime
import random

import torch
import numpy as np


class Logger(object):
    def __init__(self, filename='./logs/' + datetime.datetime.now().strftime('%Y-%m-%d %H%M%S.%f') + '.txt', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'w+')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


class BestModelSaver:
    def __init__(self):
        self.best_valid_acc = 0
        self.best_valid_auc = 0
        self.best_valid_f1 = 0
        self.best_valid_acc_epoch = 0
        self.best_valid_auc_epoch = 0
        self.best_valid_f1_epoch = 0


    def update(self, valid_acc, valid_auc, valid_f1, current_epoch):

        if valid_acc >= self.best_valid_acc:
            self.best_valid_acc = valid_acc
            self.best_valid_acc_epoch = current_epoch
        if valid_auc >= self.best_valid_auc:
            self.best_valid_auc = valid_auc
            self.best_valid_auc_epoch = current_epoch
        if valid_f1 >= self.best_valid_f1:
            self.best_valid_f1 = valid_f1
            self.best_valid_f1_epoch = current_epoch


def fix_random_seeds(seed=None):
    """
    Fix random seeds.
    """
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    print("Fix Random Seeds:", seed)


def merge_config_to_args(args, cfg):
    # Data
    args.he_feature_root = cfg.DATA.HE_FEATURE_ROOT
    args.ihc_feature_root = cfg.DATA.IHC_FEATURE_ROOT
    args.he_train_valid_csv = cfg.DATA.HE_TRAIN_VALID_CSV
    args.ihc_train_valid_csv = cfg.DATA.IHC_TRAIN_VALID_CSV
    args.he_test_csv = cfg.DATA.HE_TEST_CSV
    args.ihc_test_csv = cfg.DATA.IHC_TEST_CSV

    # Model
    args.feat_dim = cfg.MODEL.FEATURE_DIM
    args.num_classes = cfg.MODEL.NUM_CLASSES
    args.select_k = cfg.MODEL.SELECT_K
    args.kl_weight = cfg.MODEL.KL_WEIGHT
    args.return_atte = cfg.MODEL.RETURN_ATTE

    # TRAIN
    args.mscp = cfg.TRAIN.MSCP
    args.batch_size = cfg.TRAIN.BATCH_SIZE
    args.workers = cfg.TRAIN.WORKERS
    args.lr = cfg.TRAIN.LR
    args.weight_decay = cfg.TRAIN.WEIGHT_DECAY
    args.max_epoch = cfg.TRAIN.MAX_EPOCH
    args.weights_save_path = cfg.TRAIN.WEIGHTS_SAVE_PATH
