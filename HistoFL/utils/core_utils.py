import numpy as np
import torch
import pickle
import pdb
from utils.utils import *
import copy
import os
from datasets.dataset_generic import save_splits
from sklearn.metrics import roc_auc_score
from models.model_attention_mil import MIL_Attention_fc

from utils.fl_utils import sync_models, federated_averging

def split_data_by_institute(df, train_val_test_ratio, random_state=42, split_root=None):
    # 初始化一个新的列用于保存划分结果
    df['split'] = None
    
    # 遍历每个 institute 进行划分
    for institute in df['institute'].unique():
        institute_data = df[df['institute'] == institute]
        
        # 获取 unique 的 case_id 和对应的 diagnosis_label
        case_ids = institute_data['case_id'].unique()
        labels = institute_data.groupby('case_id')['diagnosis_label'].first()  # 获取每个 case 的 label
        
        # 按照 train_val_test_ratio 进行 stratified 划分
        train_size, val_size, test_size = train_val_test_ratio
        
        # Step 1: 先划分出训练集和临时集 (validation + test)
        train_case_ids, temp_case_ids, _, temp_labels = train_test_split(
            case_ids, labels, test_size=1-train_size, stratify=labels, random_state=random_state)
        
        # Step 2: 从临时集中再划分验证集和测试集
        val_case_ids, test_case_ids = train_test_split(
            temp_case_ids, test_size=test_size/(val_size+test_size), stratify=temp_labels, random_state=random_state)
        
        # 将划分结果写回 DataFrame
        df.loc[df['case_id'].isin(train_case_ids), 'split'] = 'train'
        df.loc[df['case_id'].isin(val_case_ids), 'split'] = 'val'
        df.loc[df['case_id'].isin(test_case_ids), 'split'] = 'test'
    df.to_csv(split_root,index=False)
    
    return df

class Accuracy_Logger(object):
    """Accuracy logger"""
    def __init__(self, n_classes):
        super(Accuracy_Logger, self).__init__()
        self.n_classes = n_classes
        self.initialize()

    def initialize(self):
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
    
    def log(self, Y_hat, Y):
        Y_hat = int(Y_hat)
        Y = int(Y)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat == Y)

    def log_batch(self, count, correct, c):
        self.data[c]["count"] += count
        self.data[c]["correct"] += correct
    
    def get_summary(self, c):
        count = self.data[c]["count"] 
        correct = self.data[c]["correct"]
        
        if count == 0: 
            acc = None
        else:
            acc = float(correct) / count
        
        return acc, correct, count

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=20, stop_epoch=50, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf

    def __call__(self, epoch, val_loss, model, ckpt_name = 'checkpoint.pt'):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss

def train_fl(datasets, cur, args, log=None):
    """   
        train for a single fold
    """
    # number of institutions
    # if log is not None:
    #     print = log.info
    print('\nTraining Fold {}!'.format(cur))
    writer_dir = os.path.join(args.results_dir, str(cur))
    if not os.path.isdir(writer_dir):
        os.mkdir(writer_dir)

    from tensorboardX import SummaryWriter
    writer = SummaryWriter(writer_dir, flush_secs=15)


    print('\nInit train/val/test splits...', end=' ')
    # train_splits, val_split, test_split = datasets
    train_splits, val_splits, test_splits = datasets
    num_insti = len(train_splits)
    print("num-insti：", num_insti)
    # save_splits(datasets, ['train', 'val', 'test'], os.path.join(args.results_dir, 'splits_{}.csv'.format(cur)))
    print('Done!')
    # for idx in range(num_insti):
    #     print("Worker_{} Training on {} samples".format(idx,len(train_splits[idx])))
    # print("Validating on {} samples".format(len(val_split)))
    # print("Testing on {} samples".format(len(test_split)))
    for idx in range(num_insti):
        print("Worker_{} Training on {} samples".format(idx,len(train_splits[idx])))
        print("Worker_{} Validating on {} samples".format(idx,len(val_splits[idx])))
        print("Worker_{} Testing on {} samples".format(idx,len(test_splits[idx])))

    print('\nInit loss function...', end=' ')
    loss_fn = nn.CrossEntropyLoss()
    # loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([0.01,0.99]).cuda()) # weight=torch.tensor([0.1,0.9]).cuda()
    print('Done!')
    
    print('\nInit Model...', end=' ')
    model_dict = {"dropout": args.drop_out, 'n_classes': args.n_classes}
    if args.model_type =='attention_mil':
        if args.model_size is not None:
            model_dict.update({"size_arg": args.model_size})
        
        model = MIL_Attention_fc(**model_dict)
        worker_models =[MIL_Attention_fc(**model_dict) for idx in range(num_insti)]

    
    else: # args.model_type == 'mil'
        raise NotImplementedError
    
    sync_models(model, worker_models)   
    device_counts = torch.cuda.device_count()
    if device_counts > 1:
        device_ids = [idx % device_counts for idx in range(num_insti)]
    else:
        device_ids = [0]*num_insti
    
    model.relocate(device_id=0)
    for idx in range(num_insti):
        worker_models[idx].relocate(device_id=device_ids[idx])

    print('Done!')
    print_network(model)
    print('\nInit optimizer...', end=' ')
    worker_optims = [get_optim(worker_models[i], args) for i in range(num_insti)]
    print('Done!')
    
    print('\nInit Loaders...', end=' ')
    train_loaders = []
    # ------add----
    val_loaders, test_loaders = [],[]
    # --------------
    for idx in range(num_insti):
        train_loaders.append(get_split_loader(train_splits[idx], training=True, testing = args.testing, 
                                              weighted = args.weighted_sample))
        val_loaders.append(get_split_loader(val_splits[idx], testing = args.testing))
        test_loaders.append(get_split_loader(test_splits[idx], testing = args.testing))
    # val_loader = get_split_loader(val_split, testing = args.testing)
    # test_loader = get_split_loader(test_split, testing = args.testing)

    if args.weighted_fl_avg:
        weights = np.array([len(train_loader) for train_loader in train_loaders]) 
        weights = weights / weights.sum()
    else:
        weights = None

    print('Done!')

    print('\nSetup EarlyStopping...', end=' ')
    if args.early_stopping:
        early_stopping = EarlyStopping(patience = 20, stop_epoch= 35, verbose = True)

    else:
        early_stopping = None
    print('Done!')

    for epoch in range(args.max_epochs):        
        train_loop_fl(epoch, model, worker_models, train_loaders, worker_optims, 
                     args.n_classes, writer, loss_fn, log)

        if (epoch + 1) % args.E == 0:
            print('federated averging...')
            model, worker_models = federated_averging(model, worker_models, args.noise_level, weights)
            sync_models(model, worker_models)  

        stop = validate(cur, epoch, model, val_loaders, args.n_classes, 
                early_stopping, writer, loss_fn, args.results_dir,log)
        
        # --------------add--------------
        results_dict, test_error, test_auc, acc_logger = summary(model, test_loaders, args.n_classes, log)
        print('Mid{} Test error: {:.4f}, ROC AUC: {:.4f}'.format(epoch, test_error, test_auc))
        
        if stop: 
            break

    if args.early_stopping:
        model.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur))))
    else:
        torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))

    _, val_error, val_auc, _= summary(model, val_loaders, args.n_classes, log)
    print('Final Val error: {:.4f}, ROC AUC: {:.4f}'.format(val_error, val_auc))

    results_dict, test_error, test_auc, acc_logger = summary(model, test_loaders, args.n_classes, log)
    print('Final Test error: {:.4f}, ROC AUC: {:.4f}'.format(test_error, test_auc))
    for i in range(args.n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('Final class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))

        if writer:
            writer.add_scalar('final/test_class_{}_acc'.format(i), acc, 0)
    if writer:
        writer.add_scalar('final/val_error', val_error, 0)
        writer.add_scalar('final/val_auc', val_auc, 0)
        writer.add_scalar('final/test_error', test_error, 0)
        writer.add_scalar('final/test_auc', test_auc, 0)
    
    writer.close()
    return results_dict, test_auc, val_auc, 1-test_error, 1-val_error 

    # results_dicts = {}
    # test_aucs, val_aucs, test_errors, val_errors =[], [], [], []  
    # for idx in range(num_insti):
    #     _, val_error, val_auc, _= summary(model, val_loaders[idx], args.n_classes)
    #     print('Val error: {:.4f}, ROC AUC: {:.4f}'.format(val_error, val_auc))
    #     val_aucs.append(val_auc)
    #     val_errors.append(val_error)
        
    #     results_dict, test_error, test_auc, acc_logger = summary(model, test_loaders[idx], args.n_classes)
    #     results_dicts.append(results_dict)
    #     test_aucs.append(test_auc)
    #     test_errors.append(test_error)

    #     print('Test error: {:.4f}, ROC AUC: {:.4f}'.format(test_error, test_auc))

    #     for i in range(args.n_classes):
    #         acc, correct, count = acc_logger.get_summary(i)
    #         print('institute{}, class {}: acc {}, correct {}/{}'.format(idx, i, acc, correct, count))

    #         if writer:
    #             writer.add_scalar('institute{}, final/test_class_{}_acc'.format(idx, i), acc, 0)

    #     if writer:
    #         writer.add_scalar('institute{}, final/val_error'.format(idx), val_error, 0)
    #         writer.add_scalar('institute{}, final/val_auc'.format(idx), val_auc, 0)
    #         writer.add_scalar('institute{}, final/test_error'.format(idx), test_error, 0)
    #         writer.add_scalar('institute{}, final/test_auc'.format(idx), test_auc, 0)
    
    # writer.close()
    # return results_dicts, test_aucs, val_aucs, 1-test_errors, 1-val_errors 


def train(datasets, cur, args, log=None):
    """   
        train for a single fold
    """
    # if log is not None:
    #     print = log.info
    print('\nTraining Fold {}!'.format(cur))
    writer_dir = os.path.join(args.results_dir, str(cur))
    if not os.path.isdir(writer_dir):
        os.mkdir(writer_dir)
    
    from tensorboardX import SummaryWriter
    writer = SummaryWriter(writer_dir, flush_secs=15)

    print('\nInit train/val/test splits...')
    train_split, val_split, test_split = datasets
    # print("okko", datasets[0])
    save_splits(datasets, ['train', 'val', 'test'], os.path.join(args.results_dir, 'splits_{}.csv'.format(cur)))
    print('Done!')
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))
    print("Testing on {} samples".format(len(test_split)))

    print('\nInit loss function...', end=' ')
    loss_fn = nn.CrossEntropyLoss()
    print('Done!')
    
    print('\nInit Model...', end=' ')
    model_dict = {"dropout": args.drop_out, 'n_classes': args.n_classes}
    if args.model_type =='attention_mil':
        if args.model_size is not None:
            model_dict.update({"size_arg": args.model_size})
        model = MIL_Attention_fc(**model_dict)
    
    else: 
        raise NotImplementedError
    
    model.relocate()
    print('Done!')
    print_network(model)

    print('\nInit optimizer ...', end=' ')
    optimizer = get_optim(model, args)
    print('Done!')
    
    print('\nInit Loaders...', end=' ')
    train_loader = get_split_loader(train_split, training=True, testing = args.testing, weighted = args.weighted_sample)
    val_loader = get_split_loader(val_split,  testing = args.testing)
    test_loader = get_split_loader(test_split, testing = args.testing)
    print('Done!')

    print('\nSetup EarlyStopping...', end=' ')
    if args.early_stopping:
        early_stopping = EarlyStopping(patience = 20, stop_epoch=35, verbose = True)

    else:
        early_stopping = None
    print('Done!')

    for epoch in range(args.max_epochs):   
        #  训练模型
        train_loop(epoch, model, train_loader, optimizer, args.n_classes, writer, loss_fn,log)
        stop = validate(cur, epoch, model, val_loader, args.n_classes, 
            early_stopping, writer, loss_fn, args.results_dir, log)
        
        results_dict, test_error, test_auc, acc_logger = summary(model, test_loader, args.n_classes, log)
        print('Mid{}, Test error: {:.4f}, ROC AUC: {:.4f}'.format(epoch, test_error, test_auc))
        
        if stop: 
            break

    if args.early_stopping:
        model.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur))))
    else:
        torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))

    _, val_error, val_auc, _= summary(model, val_loader, args.n_classes, log)
    print('Final Val error: {:.4f}, ROC AUC: {:.4f}'.format(val_error, val_auc))

    results_dict, test_error, test_auc, acc_logger = summary(model, test_loader, args.n_classes, log)
    print('Final Test error: {:.4f}, ROC AUC: {:.4f}'.format(test_error, test_auc))

    for i in range(args.n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))

        if writer and acc is not None:
            writer.add_scalar('final/test_class_{}_acc'.format(i), acc, 0)

    if writer:
        writer.add_scalar('final/val_error', val_error, 0)
        writer.add_scalar('final/val_auc', val_auc, 0)
        writer.add_scalar('final/test_error', test_error, 0)
        writer.add_scalar('final/test_auc', test_auc, 0)
    
    writer.close()
    return results_dict, test_auc, val_auc, 1-test_error, 1-val_error 

def train_loop_fl(epoch, model, worker_models, worker_loaders, worker_optims, n_classes, writer = None, loss_fn = None, log=None):
    # if log is not None:
    #         print = log.info
    num_insti = len(worker_models)    
    model.train()
    
    # for idx in range(num_insti):
    #     worker_models[idx].train()
   
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    
    train_loss = 0.
    train_error = 0.

    print('\n')
    # 2个worker_loaders
    for idx in range(len(worker_loaders)):
        # pdb.set_trace()
        if worker_models[idx].device is not None:
            model_device = worker_models[idx].device
        else:
            model_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        for batch_idx, (data, label) in enumerate(worker_loaders[idx]):
            data, label = data.to(model_device), label.to(model_device)
            logits, Y_prob, Y_hat, _, _ = worker_models[idx](data)

            acc_logger.log(Y_hat, label)
            loss = loss_fn(logits, label)
            loss_value = loss.item()
            train_loss += loss_value
            if (batch_idx + 1) % 5 == 0:
                print('batch {}, loss: {:.4f}, '.format(batch_idx, loss_value),
                      'label: {}, bag_size: {}'.format(label.item(), data.size(0)))

            error = calculate_error(Y_hat, label)
            train_error += error

            # backward pass
            loss.backward()
            # step
            worker_optims[idx].step()
            worker_optims[idx].zero_grad()

    # calculate loss and error for epoch
    train_loss /= np.sum(len(worker_loaders[i]) for i in range(num_insti))
    train_error /= np.sum(len(worker_loaders[i]) for i in range(num_insti))
    print('Epoch: {}, train_loss: {:.4f}, train_error: {:.4f}'.format(epoch, train_loss, train_error))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('!class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)

def train_loop(epoch, model, loader, optimizer, n_classes, writer = None, loss_fn = None, log=None):   
    # if log is not None:
    #     print = log.info
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    train_loss = 0.
    train_error = 0.

    print('\n')
    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        logits, Y_prob, Y_hat, _, _ = model(data)
        
        acc_logger.log(Y_hat, label)
        loss = loss_fn(logits, label)
        loss_value = loss.item()
        
        train_loss += loss_value
        if (batch_idx + 1) % 5 == 0:
            print('batch {}, loss: {:.4f}, label: {}, bag_size: {}'.format(batch_idx, loss_value, label.item(), data.size(0)))
           
        error = calculate_error(Y_hat, label)
        train_error += error
        
        # backward pass
        loss.backward()
        # step
        optimizer.step()
        optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss /= len(loader)
    train_error /= len(loader)

    print('Epoch: {}, train_loss: {:.4f}, train_error: {:.4f}'.format(epoch, train_loss, train_error))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('!class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)

   
# def validate(cur, epoch, model, loader, n_classes, early_stopping = None, writer = None, loss_fn = None, results_dir=None):
#     device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.eval()
#     acc_logger = Accuracy_Logger(n_classes=n_classes)
#     val_loss = 0.
#     val_error = 0.
    
#     prob = np.zeros((len(loader), n_classes))
#     labels = np.zeros(len(loader))

#     with torch.no_grad():
#         for batch_idx, (data, label) in enumerate(loader):
#             data, label = data.to(device), label.to(device)

#             logits, Y_prob, Y_hat, _, _ = model(data)

#             acc_logger.log(Y_hat, label)
            
#             loss = loss_fn(logits, label)

#             prob[batch_idx] = Y_prob.cpu().numpy()
#             labels[batch_idx] = label.item()
            
#             val_loss += loss.item()
#             error = calculate_error(Y_hat, label)
#             val_error += error
            

#     val_error /= len(loader)
#     val_loss /= len(loader)

#     if n_classes == 2:
#         auc = roc_auc_score(labels, prob[:, 1])
    
#     else:
#         auc = roc_auc_score(labels, prob, multi_class='ovr')
    
    
#     if writer:
#         writer.add_scalar('val/loss', val_loss, epoch)
#         writer.add_scalar('val/auc', auc, epoch)
#         writer.add_scalar('val/error', val_error, epoch)

#     print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}'.format(val_loss, val_error, auc))
#     for i in range(n_classes):
#         acc, correct, count = acc_logger.get_summary(i)
#         print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
#         if writer:
#             writer.add_scalar('val/class_{}_acc'.format(i), acc, epoch)
     

#     if early_stopping:
#         assert results_dir
#         early_stopping(epoch, val_loss, model, ckpt_name = os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        
#         if early_stopping.early_stop:
#             print("Early stopping")
#             return True

#     return False


# def summary(model, loader, n_classes):
#     device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     acc_logger = Accuracy_Logger(n_classes=n_classes)
#     model.eval()
#     test_loss = 0.
#     test_error = 0.

#     all_probs = np.zeros((len(loader), n_classes))
#     all_labels = np.zeros(len(loader))

#     slide_ids = loader.dataset.slide_data['slide_id']
#     patient_results = {}

#     for batch_idx, (data, label) in enumerate(loader):
#         data, label = data.to(device), label.to(device)
#         slide_id = slide_ids.iloc[batch_idx]
#         with torch.no_grad():
#             logits, Y_prob, Y_hat, _, _ = model(data)

#         acc_logger.log(Y_hat, label)
#         probs = Y_prob.cpu().numpy()
#         all_probs[batch_idx] = probs
#         all_labels[batch_idx] = label.item()
        
#         patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label.item()}})
#         error = calculate_error(Y_hat, label)
#         test_error += error

#     test_error /= len(loader)

#     if n_classes == 2:
#         auc = roc_auc_score(all_labels, all_probs[:, 1])

#     else:
#         auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')


#     return patient_results, test_error, auc, acc_logger


def validate(cur,epoch, model, loaders, n_classes, early_stopping=None, writer=None, loss_fn=None, results_dir=None, log=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    if log is not None:
        print = log.info
    

    total_val_loss = 0.
    total_val_error = 0.
    
    all_prob = []
    all_labels = []

    # 如果传入的 loaders 是一个列表
    if isinstance(loaders, list):
        loader_names = ['worker_{}'.format(i) for i in range(len(loaders))]
    else:
        loaders = [loaders]
        loader_names = ['worker_0']

    for i, loader in enumerate(loaders):
        loader_name = loader_names[i]

        val_loss = 0.
        val_error = 0.
        acc_logger = Accuracy_Logger(n_classes=n_classes)
        prob = np.zeros((len(loader), n_classes))
        labels = np.zeros(len(loader))

        with torch.no_grad():
            for batch_idx, (data, label) in enumerate(loader):
                data, label = data.to(device), label.to(device)

                logits, Y_prob, Y_hat, _, _ = model(data)

                acc_logger.log(Y_hat, label)

                loss = loss_fn(logits, label)

                prob[batch_idx] = Y_prob.cpu().numpy()
                labels[batch_idx] = label.item()

                val_loss += loss.item()
                error = calculate_error(Y_hat, label)
                val_error += error

        val_error /= len(loader)
        val_loss /= len(loader)

        if n_classes == 2:
            auc = roc_auc_score(labels, prob[:, 1])
        else:
            auc = roc_auc_score(labels, prob, multi_class='ovr')

        # 累加每个loader的prob和label，用于后面计算总体的auc
        all_prob.append(prob)
        all_labels.append(labels)

        print('\n{}: val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}'.format(loader_name, val_loss, val_error, auc))
        for j in range(n_classes):
            acc, correct, count = acc_logger.get_summary(j)
            print('{} - class {}: acc {}, correct {}/{}'.format(loader_name, j, acc, correct, count))
            if writer:
                writer.add_scalar('val/{}/class_{}_acc'.format(loader_name, j), acc, epoch)

        total_val_loss += val_loss
        total_val_error += val_error

    # 计算所有loader的总val_loss和val_error的平均值
    total_val_loss /= len(loaders)
    total_val_error /= len(loaders)

    # 汇总所有loader的prob和label，计算总的auc
    all_prob = np.vstack(all_prob)
    all_labels = np.concatenate(all_labels)

    if n_classes == 2:
        total_auc = roc_auc_score(all_labels, all_prob[:, 1])
    else:
        total_auc = roc_auc_score(all_labels, all_prob, multi_class='ovr')

    # 打印并记录总体的val_loss, val_error, auc
    print('\nTotal: val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}'.format(total_val_loss, total_val_error, total_auc))
    if writer:
        writer.add_scalar('val/total_loss', total_val_loss, epoch)
        writer.add_scalar('val/total_auc', total_auc, epoch)
        writer.add_scalar('val/total_error', total_val_error, epoch)



    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss, model, ckpt_name = os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))

        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False


def summary(model, loaders, n_classes, log):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if log is not None:
        print = log.info
    model.eval()
    
    total_test_error = 0.
    all_probs = []
    all_labels = []
    patient_results = {}

    # 如果传入的 loaders 是一个列表
    if isinstance(loaders, list):
        loader_names = ['loader_{}'.format(i) for i in range(len(loaders))]
    else:
        loaders = [loaders]
        loader_names = ['loader_0']
    
    for i, loader in enumerate(loaders):
        loader_name = loader_names[i]
        test_error = 0.
        acc_logger = Accuracy_Logger(n_classes=n_classes)
        loader_probs = []
        loader_labels = []
        slide_ids = loader.dataset.slide_data['slide_id']

        for batch_idx, (data, label) in enumerate(loader):
            data, label = data.to(device), label.to(device)
            
            # Ensure correct indexing for slide_ids in case of batch size > 1
            slide_id_batch = slide_ids.iloc[batch_idx * loader.batch_size : (batch_idx + 1) * loader.batch_size]
            
            with torch.no_grad():
                logits, Y_prob, Y_hat, _, _ = model(data)

            acc_logger.log(Y_hat, label)

            # Move batch of probabilities and labels to CPU and numpy arrays
            probs = Y_prob.cpu().numpy()
            labels = label.cpu().numpy()

            loader_probs.append(probs)
            loader_labels.append(labels)

            # Store results for each slide in the batch
            for j, slide_id in enumerate(slide_id_batch):
                patient_results.update({
                    slide_id: {
                        'slide_id': np.array(slide_id),
                        'prob': probs[j],
                        'label': labels[j]
                    }
                })

            error = calculate_error(Y_hat, label)
            test_error += error

        # Average the test error for the current loader
        test_error /= len(loader)
        total_test_error += test_error

        # Concatenate all probabilities and labels for the current loader
        loader_probs = np.vstack(loader_probs)
        loader_labels = np.concatenate(loader_labels)

        # Calculate AUC based on the number of classes
        if n_classes == 2:
            auc = roc_auc_score(loader_labels, loader_probs[:, 1])
        else:
            auc = roc_auc_score(loader_labels, loader_probs, multi_class='ovr')

        print(f'\n{loader_name}: test_error: {test_error:.4f}, auc: {auc:.4f}')
        for j in range(n_classes):
            acc, correct, count = acc_logger.get_summary(j)
            print(f'{loader_name} - class {j}: acc {acc}, correct {correct}/{count}')

        # Accumulate all probs and labels across loaders for overall evaluation
        all_probs.append(loader_probs)
        all_labels.append(loader_labels)

    # 汇总所有 loader 的 prob 和 labels
    all_probs = np.vstack(all_probs)
    all_labels = np.concatenate(all_labels)

    # Calculate total AUC
    if n_classes == 2:
        total_auc = roc_auc_score(all_labels, all_probs[:, 1])
    else:
        total_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')

    # Average the test error across all loaders
    total_test_error /= len(loaders)

    print(f'\nTotal: test_error: {total_test_error:.4f}, auc: {total_auc:.4f}')
    return patient_results, total_test_error, total_auc, acc_logger
