from __future__ import print_function

import argparse
import pdb
import os
import math

# internal imports
from utils.file_utils import save_pkl, load_pkl
from utils.utils import *
from utils.core_utils import train, train_fl 
from datasets.dataset_generic import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset

# pytorch imports
import torch
from torch.utils.data import DataLoader, sampler
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
import numpy as np
import logging

def main(args,log):
    # create results directory if necessary
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)

    if args.k_start == -1:
        start = 0
    else:
        start = args.k_start
    if args.k_end == -1:
        end = args.k
    else:
        end = args.k_end

    all_test_auc = []
    all_val_auc = []
    all_test_acc = []
    all_val_acc = []
    folds = np.arange(start, end)
    for i in folds:
        seed_torch(args.seed)
        # train_datasets, val_dataset, test_dataset = dataset.return_splits(from_id=False,
        #         csv_path='{}/splits_{}.csv'.format(args.split_dir, i), no_fl=args.no_fl)
        train_datasets, val_datasets, test_datasets = dataset.return_splits(from_id=False,
                csv_path='{}/splits_{}.csv'.format(args.split_dir, i), no_fl=args.no_fl)
        print("train长度：",len(train_datasets))
        print("val长度：",len(val_datasets))
        print("test长度：",len(test_datasets))

        if len(train_datasets)>1:
            # for idx in range(len(train_datasets)):
            #     print("worker_{} training on {} samples".format(idx,len(train_datasets[idx])))
            # print('validation: {}, testing: {}'.format(len(val_dataset), len(test_dataset)))
            # datasets = (train_datasets, val_dataset, test_dataset)
            # results, test_auc, val_auc, test_acc, val_acc  = train_fl(datasets, i, args)
            for idx in range(len(train_datasets)):
                print("worker_{} training on {} samples".format(idx,len(train_datasets[idx])))
                print("worker_{} validation on {} samples, testing on {} samples".format(idx, len(val_datasets[idx]), len(test_datasets[idx])))
            datasets = (train_datasets, val_datasets, test_datasets)
            results, test_auc, val_auc, test_acc, val_acc  = train_fl(datasets, i, args, log=log)
        else:
            # train_dataset = train_datasets[0] 
            # print('training: {}, validation: {}, testing: {}'.format(len(train_dataset), len(val_dataset), len(test_dataset)))
            # datasets = (train_dataset, val_dataset, test_dataset)
            # results, test_auc, val_auc, test_acc, val_acc  = train(datasets, i, args)
            train_dataset = train_datasets[0] 
            val_dataset = val_datasets[0] 
            test_dataset = test_datasets[0] 
            print('training: {}, validation: {}, testing: {}'.format(len(train_dataset), len(val_datasets), len(test_datasets)))
            datasets = (train_dataset, val_dataset, test_dataset)
            
            results, test_auc, val_auc, test_acc, val_acc  = train(datasets, i, args,log=log)
        
        all_test_auc.append(test_auc)
        all_val_auc.append(val_auc)
        all_test_acc.append(test_acc)
        all_val_acc.append(val_acc)
        #write results to pkl
        filename = os.path.join(args.results_dir, 'split_{}_results.pkl'.format(i))
        save_pkl(filename, results)

    final_df = pd.DataFrame({'folds': folds, 'test_auc': all_test_auc, 
        'val_auc': all_val_auc, 'test_acc': all_test_acc, 'val_acc' : all_val_acc})

    if len(folds) != args.k:
        save_name = 'summary_partial_{}_{}.csv'.format(start, end)
    else:
        save_name = 'summary.csv'
    final_df.to_csv(os.path.join(args.results_dir, save_name))

# Training settings
parser = argparse.ArgumentParser(description='Configurations for WSI Training')
parser.add_argument('--data_root_dir', type=str, default= 'DATA_ROOT_DIR',
                    help='data directory')
parser.add_argument('--max_epochs', type=int, default=50,
                    help='maximum number of epochs to train')
parser.add_argument('--lr', type=float, default=2e-4,
                    help='learning rate (default: 0.0002)')
parser.add_argument('--noise_level', type=float, default=0,
                    help='noise level added on the shared weights in federated learning (default: 0)')
parser.add_argument('--reg', type=float, default=1e-5,
                    help='weight decay (default: 1e-5)')
parser.add_argument('--seed', type=int, default=1, 
                    help='random seed for reproducible experiment (default: 1)')
parser.add_argument('--k', type=int, default=5, help='number of folds (default: 5)')
parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: -1, last fold)')
parser.add_argument('--k_end', type=int, default=-1, help='end fold (default: -1, first fold)')
#  help='results directory (default: ./results)'
parser.add_argument('--results_dir', default='./results/no_weight', help='权重5/4幂次方')
parser.add_argument('--split_dir', type=str, default='fl_classification', 
                    help='manually specify the set of splits to use (default: None)')
parser.add_argument('--opt', type=str, choices = ['adam', 'sgd'], default='adam')
parser.add_argument('--exp_code', type=str, help='experiment code for saving results')
parser.add_argument('--weighted_sample', action='store_true', default=False, help='enable weighted sampling')
parser.add_argument('--testing', default=False, help='less testing or not')
parser.add_argument('--task', default='classification', type=str)
parser.add_argument('--inst_name', type=str, default=None, help='name of institution to use')
parser.add_argument('--weighted_fl_avg', action='store_true', default=False, help='weight model weights by support during FedAvg update')
parser.add_argument('--no_fl', action='store_true', default=False, help='train on centralized data')
parser.add_argument('--E', type=int, default=1, help='communication_freq')
args = parser.parse_args()
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_torch(seed=7):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        print("I'm using GPU!!!")
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch(args.seed)

args.drop_out=True
args.early_stopping=True
args.model_type='attention_mil'
args.model_size='small'

settings = {'num_splits': args.k, 
            'k_start': args.k_start,
            'k_end': args.k_end,
            'task': args.task,
            'max_epochs': args.max_epochs, 
            'results_dir': args.results_dir, 
            'lr': args.lr,
            'experiment': args.exp_code,
            'reg': args.reg,
            'seed': args.seed,
            'model_type': args.model_type,
            'model_size': args.model_size,
            "use_drop_out": args.drop_out,
            'weighted_sample': args.weighted_sample,
            'E': args.E,
            'opt': args.opt,
            'testing':args.testing}

if args.inst_name is not None:
    settings.update({'inst_name':args.inst_name})

else:
    settings.update({'noise_level': args.noise_level,
                     'weighted_fl_avg': args.weighted_fl_avg,
                     'no_fl': args.no_fl})


print('\nLoad Dataset')

if args.task == 'classification':
    args.n_classes=2
    # dataset_csv/classification_hrd_dataset_fl.csv
    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/BRCA_fl_data.csv',
                            data_dir= os.path.join(args.data_root_dir, 'classification_features_dir'),
                            shuffle = False,
                            seed = args.seed,
                            print_info = True,
                            label_dict = {'class_0':0, 'class_1':1},
                            label_col = 'censorship',
                            inst = args.inst_name,
                            patient_strat= False)
# if args.task == 'classification':
#     args.n_classes=2
#    # dataset_csv/classification_hrd_dataset.csv
#     dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/BRCA_nofl_data.csv',
#                             data_dir= os.path.join(args.data_root_dir, 'classification_features_dir'),
#                             shuffle = False,
#                             seed = args.seed,
#                             print_info = True,
#                             label_dict = {'class_0':0, 'class_1':1},
#                             label_col = 'censorship',
#                             inst = args.inst_name,
#                             patient_strat= False)
else:
    raise NotImplementedError
    
if not os.path.isdir(args.results_dir):
    os.mkdir(args.results_dir)

args.results_dir = os.path.join(args.results_dir, str(args.exp_code) + '_s{}'.format(args.seed))
if not os.path.isdir(args.results_dir):
    os.mkdir(args.results_dir)
# ------------------设置日志格式--------------------
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# 创建 logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # 设置日志级别为INFO或其他级别

# 创建日志文件处理器
log_file = os.path.join(args.results_dir, "printing.log")  # 这里可以指定日志文件路径
file_handler = logging.FileHandler(log_file)
file_handler.setFormatter(log_formatter)

# 创建控制台处理器（可选，输出到控制台）
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)

# 将处理器添加到 logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)
# print = logger.info
print('Load Dataset')
# -----------------------------------------------------

if args.split_dir is None:
    args.split_dir = os.path.join('splits', args.task)
else:
    args.split_dir = os.path.join('splits', args.split_dir)
assert os.path.isdir(args.split_dir)

settings.update({'split_dir': args.split_dir})


# with open(args.results_dir + '/experiment_{}.txt'.format(args.exp_code), 'w') as f:
#     print(settings, file=f)
# f.close()

print("################# Settings ###################")
for key, val in settings.items():
    print("{}:  {}".format(key, val))        

if __name__ == "__main__":
    results = main(args, log=logger)
    print("finished!")
    print("end script")


