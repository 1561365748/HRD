from cProfile import label
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import numpy as np 

import sklearn.metrics as metrics
from utils.losses import FocalLoss



class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label



class LocalUpdate(object):
    def __init__(self, args, dataset, idxs):
        self.args = args


        self.ldr_train, self.ldr_test = self.train_test(dataset, list(idxs))

    def get_loss(self):
        if self.args.loss_type == 'CE':
            return nn.CrossEntropyLoss()
        elif self.args.loss_type == 'focal':
            return FocalLoss(gamma=1).cuda(self.args.gpu)



    def train_test(self, dataset, idxs):

        train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        test = DataLoader(dataset, batch_size=128)
        return train, test

    def update_weights(self, net, seed, net_glob, epoch, mu=1, lr=None):
        net_glob = net_glob

        net.train()

        if lr is None:
            optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        else:
            optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=self.args.momentum)

        epoch_loss = []
        for iter in range(epoch):
            batch_loss = []

            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)

                labels = labels.long()
                net.zero_grad()
                logits = net(images)
                criterion = self.get_loss()
                loss = criterion(logits, labels)               
                '''
                这段代码的主要目的是在训练过程中计算全局模型和当前模型之间的权重差异，
                并将其作为正则项加到损失函数中，以便在优化过程中考虑这个差异。
                这样可以在一定程度上防止模型参数的剧烈变化，从而提高模型的稳定性。
                '''
                if self.args.beta > 0:
                    if batch_idx > 0:
                        w_diff = torch.tensor(0.).to(self.args.device)
                        for w, w_t in zip(net_glob.parameters(), net.parameters()):
                            # 使用 zip 函数将 net_glob 和 net 的参数配对，然后计算每对参数之间的欧几里得距离的平方和。
                            # 最后，取平方根得到总的权重差异 w_diff
                            w_diff += torch.pow(torch.norm(w - w_t), 2)
                        w_diff = torch.sqrt(w_diff)
                        # 将计算得到的权重差异乘以 self.args.beta 和 mu，并加到损失 loss 中。
                        loss += self.args.beta * mu * w_diff

                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def update_weights_pid(self, net, seed, epoch, GBA_Loss, GBA_Layer, mu=1, lr=None):
        '''
        
        '''
        # net=copy.deepcopy(g_backbone).to(args.device),
        # 损失函数GBA_Loss = g_pid_losses[idx]
        # 线性层GBA_Layer = g_linears[idx]
        # 正则化参数 mu
        net.train()
        GBA_Layer.train()

        if lr is None:
            backbone_optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        else:
            backbone_optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=self.args.momentum)

        epoch_loss = []
        for iter in range(epoch):
            batch_loss = []


            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                # 转换为长整型张量
                labels = labels.long()




                net.zero_grad()
                # 将图像输入模型 net，获取特征 feat。
                feat = net(images)
                # 将特征 feat 输入线性层 GBA_Layer，获取输出 logits。
                logits = GBA_Layer(feat)
                # 计算损失函数
                loss = GBA_Loss(logits, labels) 
                loss.backward()
                backbone_optimizer.step()
                

                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        # 返回更新后的模型权重 net.state_dict()、线性层权重 GBA_Layer.state_dict() 和平均训练损失。
        return net.state_dict(), GBA_Layer.state_dict(), sum(epoch_loss) / len(epoch_loss)



    def update_weights_gasp_grad(self, net, seed, net_glob, client_id, epoch, gradBag, mu=1, lr=None):
        hookObj,  gradAnalysor = gradBag.get_client(client_id)
        net_glob = net_glob
        net.train()

        if lr is None:
            optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        else:
            optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=self.args.momentum)

        epoch_loss = []
        for iter in range(epoch):
            batch_loss = []

            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)

                labels = labels.long()
                net.zero_grad()
                logits = net(images)
                criterion = self.get_loss()
                loss = criterion(logits, labels)               
                hook_handle = logits.register_hook(hookObj.hook_func_tensor) 
                if self.args.beta > 0:
                    if batch_idx > 0:
                        w_diff = torch.tensor(0.).to(self.args.device)
                        for w, w_t in zip(net_glob.parameters(), net.parameters()):
                            w_diff += torch.pow(torch.norm(w - w_t), 2)
                        w_diff = torch.sqrt(w_diff)
                        loss += self.args.beta * mu * w_diff
                loss.backward()
            
                if hookObj.has_gradient():
                    gradAnalysor.update(hookObj.get_gradient(), labels.cpu().numpy().tolist()) 
                optimizer.step()
                hook_handle.remove()
                batch_loss.append(loss.item())
                gradAnalysor.print_for_debug()

            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        gradBag.load_in(client_id, hookObj, gradAnalysor)
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss), gradBag

def calculate_metrics(pred_np, seg_np):


    b = len(pred_np)
    all_f1 = []
    all_sensitivity = []
    all_specificity = []
    all_ppv = []
    all_npv = []
    for i in range(b):
        
        f1 = metrics.f1_score(seg_np[i], pred_np[i], average='macro')




























        all_f1.append(f1)





    return all_f1




