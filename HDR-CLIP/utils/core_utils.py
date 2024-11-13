import numpy as np
import torch
from utils.participator import *
import pdb
from utils.utils import *
import copy
import os
from datasets.dataset_generic import *
from sklearn.metrics import roc_auc_score
from models.model_attention import MIL_Attention_fc
# from models.model_attention_mil import MIL_Attention_fc
from sklearn.model_selection import train_test_split
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

def train_fl(datasets, cur, args, clip_model, text_features, new_text_features, log=None):
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

    train_splits, val_splits, test_splits = datasets
    num_insti = len(train_splits)
    print("num-insti：", num_insti)
    
    print('Done!')
    
    for idx in range(num_insti):
        print("Worker_{} Training on {} samples".format(idx,len(train_splits[idx])))
        print("Worker_{} Validating on {} samples".format(idx,len(val_splits[idx])))
        print("Worker_{} Testing on {} samples".format(idx,len(test_splits[idx])))

    # print('\nInit loss function...', end=' ')
    # loss_fn = nn.CrossEntropyLoss()
    # loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([0.01,0.99]).cuda()) # weight=torch.tensor([0.1,0.9]).cuda()
    # print('Done!')
    
    print('\nInit Model...', end=' ')
    model_dict = {"dropout": args.drop_out, 'n_classes': args.n_classes}
    # if args.model_type =='attention_mil':
    #     if args.model_size is not None:
    #         model_dict.update({"size_arg": args.model_size})
        
    #     model = MIL_Attention_fc(**model_dict)
    #     worker_models =[MIL_Attention_fc(**model_dict) for idx in range(num_insti)]
    # else: # args.model_type == 'mil'
    #     raise NotImplementedError
    # 同步参数
    # sync_models(model, worker_models)   
    # device_counts = torch.cuda.device_count()

    # if device_counts > 1:
    #     device_ids = [idx % device_counts for idx in range(num_insti)]
    # else:
    #     device_ids = [0]*num_insti
    
    # model.relocate(device_id=0)
    # for idx in range(num_insti):
    #     worker_models[idx].relocate(device_id=device_ids[idx])

    # print('\nInit optimizer...', end=' ')
    # worker_optims = [get_optim(worker_models[i], args) for i in range(num_insti)]
    # print('Done!')
    
    # print('\nInit Loaders...', end=' ')
    # train_loaders = []
    # val_loaders, test_loaders = [],[]
        # for idx in range(num_insti):
    #     train_loaders.append(get_split_loader(train_splits[idx], training=True, testing = args.testing, 
    #                                           weighted = args.weighted_sample))
    #     val_loaders.append(get_split_loader(val_splits[idx], testing = args.testing))
    #     test_loaders.append(get_split_loader(test_splits[idx], testing = args.testing))
    # # val_loader = get_split_loader(val_split, testing = args.testing)
    # # test_loader = get_split_loader(test_split, testing = args.testing)

    # if args.weighted_fl_avg:
    #     weights = np.array([len(train_loader) for train_loader in train_loaders]) 
    #     weights = weights / weights.sum()
    # else:
    #     weights = None

    # print('Done!')

    # print('\nSetup EarlyStopping...', end=' ')
    # if args.early_stopping:
    #     early_stopping = EarlyStopping(patience = 20, stop_epoch= 35, verbose = True)

    # else:
    #     early_stopping = None
    # print('Done!')

    # -------------加载数据集----------------
    print('\nInit Loaders...', end=' ')
    train_loaders = []
    val_loaders, test_loaders = [],[]
    # indices2data = Clip_Indices2Dataset(train_splits, clip_datas)
    # print("indices2data_type:", indices2data)

    for idx in range(num_insti):
        train_loaders.append(get_split_loader(train_splits[idx], training=True, testing = args.testing, 
                                              weighted = args.weighted_sample))
        # train_loaders.append(get_split_loader(indices2data[idx], training=True, testing = args.testing, 
        #                                       weighted = args.weighted_sample))
        val_loaders.append(get_split_loader(val_splits[idx], testing = args.testing))
        test_loaders.append(get_split_loader(test_splits[idx], testing = args.testing))
    # -----------------------------------------
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
    # ------------加载全局模型---------------
    if args.model_type =='attention_mil':
        if args.model_size is not None:
            model_dict.update({"size_arg": args.model_size})
        
        model = Global(num_classes = args.n_classes,
                            args=args,
                            model_dict = model_dict,
                            num_of_feature=args.num_of_feature)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        temp_model = nn.Linear(512, args.n_classes).to(device)
        syn_params = temp_model.state_dict()
    else: # args.model_type == 'mil'
        raise NotImplementedError
    print('Done!')
    # print_network(model)

    # -----------------训练模型-----------------
    # for r in tqdm(range(1, args.num_rounds+1), desc='server-training'):
    for epoch in range(args.max_epochs):
        global_params = model.download_params()
        syn_feature_params = copy.deepcopy(global_params)
        for name_param in reversed(syn_feature_params):
            # 更新分类器参数
            if name_param == 'classifier.bias':
                syn_feature_params[name_param] = syn_params['bias']
            if name_param == 'classifier.weight':
                syn_feature_params[name_param] = syn_params['weight']
                break
        # -----------------局部模型-----------------
        # 初始化三个空列表，分别用于存储客户端的梯度、本地参数字典和本地数据数量。
        list_clients_gradient = []
        list_dicts_local_params = []
        list_nums_local_data = []

        for client in range(len(train_loaders)):
            data_client = train_loaders[client]
            list_nums_local_data.append(len(data_client))

            if args.model_type =='attention_mil':
                if args.model_size is not None:
                    model_dict.update({"size_arg": args.model_size})
                    device_counts = torch.cuda.device_count()
                    if device_counts > 1:
                        device_ids = [idx % device_counts for idx in range(num_insti)]
                    else:
                        device_ids = [0]*num_insti

                    # 计算当前客户端训练数据集中的类别以及各个类别的数量
                    class_counts = data_client.dataset.slide_data['label'].value_counts().to_dict()
                    class_num = [class_counts.get(i, 0) for i in range(args.n_classes)]

                    local_model = Local(data_client=data_client, device=device_ids[client],
                                class_list= class_num, model_dict=model_dict, args=args)
            else: # args.model_type == 'mil'
                raise NotImplementedError

            # compute the real feature gradients in local data
            # 计算本地数据的真实特征梯度
            '''
            计算每个类别的真实特征梯度。主要步骤包括：
            加载全局参数到本地模型。
            计算每个类别的梯度，并进行多次重复计算以获得平均梯度。
            '''
            # 公式3
            truth_gradient = local_model.compute_gradient(copy.deepcopy(syn_feature_params), args)
            list_clients_gradient.append(copy.deepcopy(truth_gradient))
            # local update 本地模型更新
            '''
            本地模型训练。主要步骤包括：
            加载全局参数到本地模型。
            进行多轮本地训练，每轮训练中使用数据增强、计算损失（交叉熵损失 +知识蒸馏损失）并进行反向传播更新模型参数。
            '''
            # 公式1,2
            local_params = local_model.local_train(args, copy.deepcopy(global_params), clip_model, text_features)
            list_dicts_local_params.append(copy.deepcopy(local_params))

        if (epoch + 1) % args.E == 0:
            # -------------实现了FedAvg算法，用于聚合本地模型参数-------------
            # 接收多个本地模型参数字典和对应的数据量列表，计算加权平均值来生成全局模型参数。
            # 公式9
            fedavg_params = model.initialize_for_model_fusion(list_dicts_local_params, list_nums_local_data)
            # 更新了特征合成网络的参数。
            # 首先加载全局参数，然后计算每个类的梯度并进行聚合，最后通过反向传播更新特征合成网络的参数
            model.update_feature_syn(args, copy.deepcopy(syn_feature_params), list_clients_gradient, new_text_features)
            # re-trained classifier
            # 重新训练特征合成网络。 
            # 使用合成特征和标签创建数据集，并使用SGD优化器训练一个线性分类器。训练完成后，更新全局参数。
            syn_params, ft_params = model.feature_re_train(copy.deepcopy(fedavg_params))
            model.syn_model.load_state_dict(copy.deepcopy(fedavg_params))

            # global evaluation
            stop = validate(cur, epoch, model, val_loaders, args.n_classes, 
                early_stopping, writer, model.optimizer_feature, args.results_dir,log)
        
            # ---------------------------------------
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



def train(datasets, clip_data, cur, args, clip_model, text_features, new_text_features, log=None):
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

    # print('\nInit loss function...', end=' ')
    # loss_fn = nn.CrossEntropyLoss()
    # print('Done!')
    
    # print('\nInit Model...', end=' ')
    # model_dict = {"dropout": args.drop_out, 'n_classes': args.n_classes}
    # if args.model_type =='attention_mil':
    #     if args.model_size is not None:
    #         model_dict.update({"size_arg": args.model_size})
    #     model = MIL_Attention_fc(**model_dict)
    
    # else: 
    #     raise NotImplementedError
    
    # model.relocate()
    # print('Done!')
    # print_network(model)

    print('\nInit Model...', end=' ')
    model_dict = {"dropout": args.drop_out, 'n_classes': args.n_classes}
    if args.model_type =='attention_mil':
        if args.model_size is not None:
            model_dict.update({"size_arg": args.model_size})
        model = Global(num_classes = args.n_classes,
                            args=args,
                            model_dict = model_dict,
                            num_of_feature=args.num_of_feature)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        temp_model = nn.Linear(512, args.n_classes).to(device)
        syn_params = temp_model.state_dict()
    else: # args.model_type == 'mil'
        raise NotImplementedError
    print('Done!')
    # print_network(model)

    # print('\nInit optimizer ...', end=' ')
    # optimizer = get_optim(model, args)
    # print('Done!')
    
    print('\nInit Loaders...', end=' ')
    indices2data = Clip_Indices2Dataset(train_split, clip_data)
    train_loader = get_split_loader(indices2data, training=True, testing = args.testing, weighted = args.weighted_sample)
    # train_loader = get_split_loader(train_split, training=True, testing = args.testing, weighted = args.weighted_sample)
    val_loader = get_split_loader(val_split,  testing = args.testing)
    test_loader = get_split_loader(test_split, testing = args.testing)
    print('Done!')

    print('\nSetup EarlyStopping...', end=' ')
    if args.early_stopping:
        early_stopping = EarlyStopping(patience = 20, stop_epoch=35, verbose = True)

    else:
        early_stopping = None
    print('Done!')
    #  训练模型
    for epoch in range(args.max_epochs):   
        global_params = model.download_params()
        syn_feature_params = copy.deepcopy(global_params)
        for name_param in reversed(syn_feature_params):
            # 更新分类器参数
            if name_param == 'classifier.bias':
                syn_feature_params[name_param] = syn_params['bias']
            if name_param == 'classifier.weight':
                syn_feature_params[name_param] = syn_params['weight']
                break
        # -----------------局部模型-----------------
        # 初始化三个空列表，分别用于存储客户端的梯度、本地参数字典和本地数据数量。
        client_gradient = []
        dict_local_params = []
        nums_local_data = []

        data_client = train_loader
        nums_local_data.append(len(data_client))   
        if args.model_type =='attention_mil':
            if args.model_size is not None:
                model_dict.update({"size_arg": args.model_size})
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
                # 计算当前客户端训练数据集中的类别以及各个类别的数量
                class_counts = data_client.dataset.slide_data['label'].value_counts().to_dict()
                class_num = [class_counts.get(i, 0) for i in range(args.n_classes)]

                local_model = Local(data_client=data_client, device= device,
                            class_list= class_num, model_dict=model_dict, args=args)
        else: # args.model_type == 'mil'
            raise NotImplementedError

        # compute the real feature gradients in local data
        # 计算本地数据的真实特征梯度
        '''
        计算每个类别的真实特征梯度。主要步骤包括：
        加载全局参数到本地模型。
        计算每个类别的梯度，并进行多次重复计算以获得平均梯度。
        '''
        # 公式3
        truth_gradient = local_model.compute_gradient(copy.deepcopy(syn_feature_params), args)
        client_gradient.append(copy.deepcopy(truth_gradient))
        # local update 本地模型更新
        '''
        本地模型训练。主要步骤包括：
        加载全局参数到本地模型。
        进行多轮本地训练，每轮训练中使用数据增强、计算损失（交叉熵损失 +知识蒸馏损失）并进行反向传播更新模型参数。
        '''
        # 公式1,2
        local_params = local_model.local_train(args, copy.deepcopy(global_params), clip_model, text_features)
        dict_local_params.append(copy.deepcopy(local_params))

        # -------------实现了FedAvg算法，用于聚合本地模型参数-------------
        # 接收多个本地模型参数字典和对应的数据量列表，计算加权平均值来生成全局模型参数。
        # 公式9
        fedavg_params = model.initialize_for_model_fusion(dict_local_params, nums_local_data)
        # 更新了特征合成网络的参数。
        # 首先加载全局参数，然后计算每个类的梯度并进行聚合，最后通过反向传播更新特征合成网络的参数
        model.update_feature_syn(args, copy.deepcopy(syn_feature_params), client_gradient, new_text_features)
        # re-trained classifier
        # 重新训练特征合成网络。 
        # 使用合成特征和标签创建数据集，并使用SGD优化器训练一个线性分类器。训练完成后，更新全局参数。
        syn_params, ft_params = model.feature_re_train(copy.deepcopy(fedavg_params))
        model.syn_model.load_state_dict(copy.deepcopy(fedavg_params))

        # global evaluation
        stop = validate(cur, epoch, model, val_loader, args.n_classes, 
            early_stopping, writer, model.optimizer_feature, args.results_dir,log)

        # ---------------------------------------
        results_dict, test_error, test_auc, acc_logger = summary(model, test_loader, args.n_classes, log)
        print('Mid{} Test error: {:.4f}, ROC AUC: {:.4f}'.format(epoch, test_error, test_auc))
        
        if stop: 
            break
        # train_loop(epoch, model, train_loader, optimizer, args.n_classes, writer, loss_fn,log)
        # stop = validate(cur, epoch, model, val_loader, args.n_classes, 
        #     early_stopping, writer, loss_fn, args.results_dir, log)
        
        # results_dict, test_error, test_auc, acc_logger = summary(model, test_loader, args.n_classes, log)
        # print('Mid{}, Test error: {:.4f}, ROC AUC: {:.4f}'.format(epoch, test_error, test_auc))
        
        # if stop: 
        #     break

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

'''
def train_loop_fl(epoch, model, worker_models, worker_loaders, worker_optims, n_classes, writer = None, loss_fn = None, log=None):
    # if log is not None:
    #         print = log.info
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
            # batch_size = data.size(0)
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
'''
def validate(cur,epoch, model, loaders, n_classes, early_stopping=None, writer=None, loss_fn=None, results_dir=None, log=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    # if log is not None:
    #     print = log.info
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
