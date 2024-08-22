# STL
import os
import gc
import copy
import sys
import argparse
import datetime
import shutil

# 3rd party library
from yacs.config import CfgNode
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sklearn.metrics import f1_score

# local library
from models.WSMCL import WSMCL
from dataset import TrainDataset, ValDataset, TestDataset
from utils.utils import merge_config_to_args, Logger, fix_random_seeds, BestModelSaver
import utils.metric as metric



def init_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def model_eval(args, test_loader, model):
    model.eval()
    correct = 0
    total = 0

    y_result = []
    pred_probs = []
    y_preds = []
    with torch.no_grad():
        for step, (he_feat, ihc_feat, target) in enumerate(test_loader):
            if len(he_feat.shape) == 3:
                he_feat = he_feat.to(args.device)
                he_feat = he_feat.squeeze(0)
            else:
                he_feat = None
            if len(ihc_feat.shape) == 3:
                ihc_feat = ihc_feat.to(args.device)
                ihc_feat = ihc_feat.squeeze(0)
            else:
                ihc_feat = None
            target = target.to(args.device)

            result = model(he_feat, ihc_feat)

            y_probs = F.softmax(result['MM_logit'], dim=1)
            y_pred = torch.argmax(result['MM_logit'], 1)

            y_result += target.cpu().tolist()
            pred_probs += y_probs.cpu().tolist()
            y_preds += y_pred.cpu().tolist()

            correct += (y_pred == target).sum().float()
            total += len(target)

    acc = (correct / total).cpu().data.numpy()

    if args.num_classes == 2:
        y_result = np.array(y_result)
        pred_probs = np.array(pred_probs)[:, 1]
        macro_auc_score = metric.macro_auc(y_result, y_score=pred_probs, multi_class=args.num_classes > 2)
        macro_F1 = f1_score(y_true=y_result, y_pred=y_preds, average='macro')
    else:
        macro_auc_score = metric.macro_auc(y_result, y_score=pred_probs, multi_class=args.num_classes > 2)
        macro_F1 = f1_score(y_true=y_result, y_pred=y_preds, average='macro')

    np.save(os.path.join(args.fold_save_path, 'target.npy'), np.array(y_result))
    np.save(os.path.join(args.fold_save_path, 'prob.npy'), np.array(pred_probs))
    np.save(os.path.join(args.fold_save_path, 'pred.npy'), np.array(y_preds))

    return acc, macro_auc_score, macro_F1


def valid(args, valid_loader, model):
    model.eval()

    correct = 0
    total = 0
    y_result = []
    pred_probs = []
    y_preds = []
    with torch.no_grad():
        for step, (he_feat, ihc_feat, target) in enumerate(valid_loader):
            if len(he_feat.shape) == 3:
                he_feat = he_feat.to(args.device)
                he_feat = he_feat.squeeze(0)
            else:
                he_feat = None
            if len(ihc_feat.shape) == 3:
                ihc_feat = ihc_feat.to(args.device)
                ihc_feat = ihc_feat.squeeze(0)
            else:
                ihc_feat = None
            target = target.to(args.device)

            result = model(he_feat, ihc_feat)
            y_probs = torch.softmax(result['MM_logit'], dim=1)
            y_pred = torch.argmax(result['MM_logit'], dim=1)
            y_preds += y_pred.cpu().tolist()
            y_result += target.cpu().tolist()
            pred_probs += y_probs.cpu().tolist()

            correct += (y_pred == target).sum().float()
            total += len(target)

    acc = (correct / total).cpu().detach().data.numpy()
    if args.num_classes == 2:
        y_result = np.array(y_result)
        pred_probs = np.array(pred_probs)[:, 1]
        auc_score = metric.macro_auc(y_result, y_score=pred_probs, multi_class=args.num_classes > 2)
        macro_f1 = f1_score(y_true=y_result, y_pred=y_preds, average='macro')
    else:
        auc_score = metric.macro_auc(np.array(y_result), y_score=np.array(pred_probs), multi_class=args.num_classes > 2)
        macro_f1 = f1_score(y_true=np.array(y_result), y_pred=np.array(y_preds), average='macro')

    return acc, auc_score, macro_f1


def train(args, model, train_loader, valid_loader, scaler):

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = torch.nn.CrossEntropyLoss()
    loss_fn.to(args.device)
    args.current_epoch = 0
    best_model_saver = BestModelSaver()

    for epoch in range(args.max_epoch):
        NUM_STEP = 5
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for step, (he_feat, ihc_feat, target) in enumerate(train_loader):
            if len(he_feat.shape) == 3:
                he_feat = he_feat.to(args.device)
                he_feat = he_feat.squeeze(0)
            else:
                he_feat = None
            if len(ihc_feat.shape) == 3:
                ihc_feat = ihc_feat.to(args.device)
                ihc_feat = ihc_feat.squeeze(0)
            else:
                ihc_feat = None
            target = target.to(args.device)

            with torch.cuda.amp.autocast():
                result = model(he_feat, ihc_feat)

                loss_ihc = loss_fn(result['IHC_logit'], target[0])
                loss_MM = loss_fn(result['MM_logit']
                                        , target[0])
                loss_c = result['c_loss']
                KL_loss_HE = F.kl_div(result['HE_logit'].softmax(dim=-1).log(),
                                      result['IHC_logit'].softmax(dim=-1), reduction='batchmean')
                total_loss = loss_MM + loss_ihc + loss_c + args.kl_weight*KL_loss_HE
                total_loss /= NUM_STEP

            scaler.scale(total_loss).backward()

            if ((step + 1) % NUM_STEP == 0) or (step + 1 == len(train_loader)):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            torch.cuda.empty_cache()
            _ = gc.collect()

            print("\rEpoch:{:d} train batch {:d}:{:d} KL_loss={:.3f} loss_ce_IHC={:.3f} loss_MM:{:.3f} "
                  "c_loss={:.3f}".format(
                epoch + 1, step + 1, len(train_loader),
                KL_loss_HE.item(), loss_ihc.item(), loss_MM.item(), loss_c.item()), end="")
        print()

        valid_acc, valid_auc, valid_f1 = valid(args, valid_loader, model)
        best_model_saver.update(valid_acc, valid_auc, valid_f1, args.current_epoch)
        print('\tValidation-Epoch: {} || valid_acc: {:.6f} || valid_auc: {:.6f} || valid_f1: {:.6f}'
              .format(args.current_epoch + 1, valid_acc, valid_auc, valid_f1))

        current_model_weight = copy.deepcopy(model.state_dict())
        torch.save(current_model_weight,
                   os.path.join(args.fold_save_path, 'epoch' + str(args.current_epoch) + '.pth'))

        args.current_epoch += 1

    shutil.copyfile(os.path.join(args.fold_save_path, 'epoch' + str(best_model_saver.best_valid_acc_epoch) + '.pth'),
                    os.path.join(args.fold_save_path, 'best_acc.pth'))
    shutil.copyfile(os.path.join(args.fold_save_path, 'epoch' + str(best_model_saver.best_valid_auc_epoch) + '.pth'),
                    os.path.join(args.fold_save_path, 'best_auc.pth'))
    shutil.copyfile(os.path.join(args.fold_save_path, 'epoch' + str(best_model_saver.best_valid_f1_epoch) + '.pth'),
                    os.path.join(args.fold_save_path, 'best_f1.pth'))
    return best_model_saver


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WSMCL script")
    parser.add_argument('--cfg', type=str, default="./configs/WSMCL_same_case_config.yaml")

    args = parser.parse_args()

    if args.cfg:
        cfg = CfgNode(new_allowed=True)
        cfg.merge_from_file(args.cfg)
        merge_config_to_args(args, cfg)

    # Save_dir init
    cfg_name = os.path.split(args.cfg)[-1]
    args.weights_save_path = os.path.join(args.weights_save_path, os.path.splitext(cfg_name)[0],
                                          datetime.datetime.now().strftime('%Y-%m-%d %H%M%S.%f'))
    os.makedirs(args.weights_save_path, exist_ok=True)
    shutil.copyfile(args.cfg, os.path.join(args.weights_save_path, cfg_name))

    sys.stdout = Logger(filename=os.path.join(args.weights_save_path,
                                              datetime.datetime.now().strftime('%Y-%m-%d %H%M%S.%f') + '.txt'))

    val_acc_5fold_model = []
    val_macro_auc_5fold_model = []
    val_macro_f1_5fold_model = []
    test_acc_5fold_model = []
    test_macro_auc_5fold_model = []
    test_macro_f1_5fold_model = []

    # Fix_random_seeds

    fix_random_seeds()
    for fold in range(5):
        # Model init

        model = WSMCL(input_dim=args.feat_dim,
                      n_classes=args.num_classes,
                      k=args.select_k,
                      return_atte=args.return_atte
        )
        if torch.cuda.is_available():
            args.device = torch.device('cuda:0')
            model = model.to(args.device)
        print("\tWSMCL feat_dim:{} n_classes:{} select_k:{} return_atte:{} kl_weight:{}"
              .format(args.feat_dim,
                            args.num_classes,
                            args.select_k,
                            args.return_atte,
                            args.kl_weight))

        args.fold_save_path = os.path.join(args.weights_save_path, 'fold' + str(fold+1))
        os.makedirs(args.fold_save_path, exist_ok=True)

        print('Training Folder: {}.\n\tData Loading...'.format(fold+1))
        train_dataset = TrainDataset(he_feature_path=args.he_feature_root,
                                     ihc_feature_path=args.ihc_feature_root,
                                     he_csv=args.he_train_valid_csv,
                                     ihc_csv=args.ihc_train_valid_csv,
                                     mscp= args.mscp,
                                     fold_k=fold+1,
                                     sample_num=None,
                                     num_classes=args.num_classes)
        valid_dataset = ValDataset(he_feature_path=args.he_feature_root,
                                     ihc_feature_path=args.ihc_feature_root,
                                     he_csv=args.he_train_valid_csv,
                                     ihc_csv=args.ihc_train_valid_csv,
                                     fold_k=fold+1,
                                     val_mode='mm',
                                     mscp=args.mscp,
                                     num_classes=args.num_classes)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.workers,pin_memory=True)
        valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                                  pin_memory=True)

        scaler = torch.cuda.amp.GradScaler()

        best_model_saver = train(args, model, train_loader, valid_loader, scaler)
        print('\t(Valid)Best ACC: {:.6f} || Best Macro_AUC: {:.6f} || Best Macro_F1: {:.6f}'
              .format(best_model_saver.best_valid_acc, best_model_saver.best_valid_auc,
                      best_model_saver.best_valid_f1))

        val_acc_5fold_model.append(best_model_saver.best_valid_acc)
        val_macro_auc_5fold_model.append(best_model_saver.best_valid_auc)
        val_macro_f1_5fold_model.append(best_model_saver.best_valid_f1)

        test_dataset = TestDataset(he_feature_path=args.he_feature_root,
                                     ihc_feature_path=args.ihc_feature_root,
                                     he_csv=args.he_test_csv,
                                     ihc_csv=args.ihc_test_csv,
                                     test_mode='mm',
                                     mscp=args.mscp,
                                     num_classes=args.num_classes)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                                 pin_memory=True)

        best_acc_model_weight = torch.load(os.path.join(args.fold_save_path, 'best_auc.pth'))
        model.load_state_dict(best_acc_model_weight)

        test_acc, test_macro_auc, test_macro_f1 = model_eval(args, test_loader, model)
        test_acc_5fold_model.append(test_acc)
        test_macro_auc_5fold_model.append(test_macro_auc)
        test_macro_f1_5fold_model.append(test_macro_f1)
        print('\t(Test)Best AUC model || ACC: {:.6f} || Macro_AUC: {:.6f} || Macro_F1: {:.6f} '
              .format(test_acc, test_macro_auc, test_macro_f1))

    print("Five-Fold-Validation:")
    print("\tVal-> Best_AUC_Model: ACC: {:.6f}±{:.6f}, Macro_AUC: {:.6f}±{:.6f}, Macro_F1: {:.6f}±{:.6f}"
          .format(np.mean(val_acc_5fold_model) * 100, np.std(val_acc_5fold_model) * 100,
                  np.mean(val_macro_auc_5fold_model) * 100, np.std(val_macro_auc_5fold_model) * 100,
                  np.mean(val_macro_f1_5fold_model) * 100, np.std(val_macro_f1_5fold_model) * 100))
    print("\tTest-> Best_AUC_Model: ACC: {:.6f}±{:.6f}, Macro_AUC: {:.6f}±{:.6}, Macro_F1: {:.6f}±{:.6}"
          .format(np.mean(test_acc_5fold_model) * 100, np.std(test_acc_5fold_model) * 100,
                  np.mean(test_macro_auc_5fold_model) * 100, np.std(test_macro_auc_5fold_model) * 100,
                  np.mean(test_macro_f1_5fold_model) * 100, np.std(test_macro_f1_5fold_model) * 100))