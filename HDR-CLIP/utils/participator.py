import os
from torchvision import datasets
from torchvision.transforms import ToTensor, transforms
from models.model_attention import MIL_Attention_fc
import numpy as np
from torch import stack, max, eq, no_grad, tensor, unsqueeze, split
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from torch.utils.data.dataloader import DataLoader

# from Model_teacher import model_dict
from torchvision.transforms import InterpolationMode
BICUBIC = InterpolationMode.BICUBIC
import torch.nn.functional as F
from utils.param_aug import DiffAugment
from utils.participator import *
from utils.losses import SupConLoss_text
from utils.Gradient_matching_loss import match_loss
from datasets.dataset_generic import TensorDataset
import clip
from tqdm import tqdm
import copy
import torch
import random
import torch.nn as nn
import time
clip_model, preprocess = clip.load('ViT-B/32', "cuda" if torch.cuda.is_available() else "cpu")
class KDLoss(nn.Module):
    '''
    Distilling the Knowledge in a Neural Network
    https://arxiv.org/pdf/1503.02531.pdf
    '''
    def __init__(self, T):
        super(KDLoss, self).__init__()
        self.T = T

    def forward(self, out_s, out_t):
        # kd = F.kl_div(F.log_softmax(out_s / self.T, dim=1),
        #               F.softmax(out_t / self.T, dim=1),
        #               reduction='none').mean(dim=0)
        kd_loss = F.kl_div(F.log_softmax(out_s/self.T, dim=1),
                        F.softmax(out_t/self.T, dim=1),
                        reduction='batchmean') * self.T * self.T

        return kd_loss

class BKD2Loss(nn.Module):

    def __init__(self, T):
        super(BKD2Loss, self).__init__()
        self.T = T

    def forward(self, out_s, out_t, weight_lamda):
        pred_t = F.softmax(out_t/self.T, dim=1)
        pred_t = pred_t * weight_lamda
        pred_t = pred_t / pred_t.sum(1)[:, None]

        kd = F.kl_div(F.log_softmax(out_s/self.T, dim=1),
                        pred_t,
                        reduction='none').mean(dim=0)
        kd_loss = kd.sum() * self.T * self.T

        return kd_loss


class Global(object):
    def __init__(self,
                 num_classes: int,
                 args,
                 model_dict,
                 num_of_feature):
        self.num_classes = num_classes
        self.fedavg_acc = []
        self.fedavg_many = []
        self.fedavg_medium = []
        self.fedavg_few = []
        self.ft_acc = []
        self.ft_many = []
        self.ft_medium = []
        self.ft_few = []
        self.num_of_feature = num_of_feature
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        self.feature_syn = torch.randn(size=(args.n_classes * self.num_of_feature, 512), dtype=torch.float,
                                       requires_grad=True, device=self.device)
        self.label_syn = torch.tensor([np.ones(self.num_of_feature) * i for i in range(args.n_classes)], dtype=torch.long,
                                      requires_grad=False, device=self.device).view(-1)  # [0,0,0, 1,1,1, ..., 9,9,9]
        self.optimizer_feature = SGD([self.feature_syn, ], lr=args.lr)  # optimizer_img for synthetic data
        self.criterion = CrossEntropyLoss().to(self.device)

        # PCL loss
        self.contras_criterion = SupConLoss_text(self.device, args.ins_temp, args.n_classes)

        self.syn_model = MIL_Attention_fc(**model_dict).to(self.device)
        self.feature_net = nn.Linear(512, args.n_classes).to(self.device)
    
    def relocate(self, device_id=None):
        if device_id is not None:
            device = 'cuda:{}'.format(device_id)
            self.syn_model.relocate(device)
        
        else:
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.syn_model.relocate(device)
       
    def update_feature_syn(self, args, global_params, list_clients_gradient, new_text_features):
        feature_net_params = self.feature_net.state_dict()
        for name_param in reversed(global_params):
            if name_param == 'classifier.bias':
                feature_net_params['bias'] = global_params[name_param]
            if name_param == 'classifier.weight':
                feature_net_params['weight'] = global_params[name_param]
                break
        self.feature_net.load_state_dict(feature_net_params)
        self.feature_net.train()
        net_global_parameters = list(self.feature_net.parameters())
        # 初始化一个字典 gw_real_all，用于存储每个类别的梯度。
        gw_real_all = {class_index: [] for class_index in range(self.num_classes)}
        # 将每个客户端的梯度添加到 gw_real_all 中。
        for gradient_one in list_clients_gradient:
            for class_num, gradient in gradient_one.items():
                gw_real_all[class_num].append(gradient)
        # 用于存储每个类别的平均梯度
        gw_real_avg = {class_index: [] for class_index in range(args.n_classes)}
        # aggregate the real feature gradients 公式4
        for i in range(args.n_classes):
            gw_real_temp = []
            list_one_class_client_gradient = gw_real_all[i]
            # 每个类别，计算其所有客户端梯度的加权平均值，并存储在 gw_real_avg 中
            if len(list_one_class_client_gradient) != 0:
                weight_temp = 1.0 / len(list_one_class_client_gradient)
                for name_param in range(2):
                    list_values_param = []
                    for one_gradient in list_one_class_client_gradient:
                        list_values_param.append(one_gradient[name_param] * weight_temp)
                    value_global_param = sum(list_values_param)
                    gw_real_temp.append(value_global_param)
                gw_real_avg[i] = gw_real_temp
        # update the federated features. 更新联邦特征 
        # 通过梯度匹配损失和原型对比学习损失将联合特征优化
        for ep in range(args.match_epoch):
            loss_feature = torch.tensor(0.0).to(self.device)
            # 遍历所有类别，计算每个类别的损失和梯度匹配损失
            for c in range(args.n_classes):
                if len(gw_real_avg[c]) != 0:
                    # 获取合成特征 feature_syn 和对应的标签 lab_syn
                    # 提取从索引 c * self.num_of_feature 到 (c + 1) * self.num_of_feature 之间的数据，重塑为一个有 self.num_of_feature 行和 512 列的二维张量。
                    feature_syn = self.feature_syn[c * self.num_of_feature:(c + 1) * self.num_of_feature].reshape((self.num_of_feature, 512))
                    lab_syn = torch.ones((self.num_of_feature,), device=self.device, dtype=torch.long) * c
                    # print("test lab_syn: ", lab_syn, lab_syn.shape)
                    # 用特征网络计算输出 output_syn
                    output_syn = self.feature_net(feature_syn)
                    # 计算交叉熵损失
                    loss_syn = self.criterion(output_syn, lab_syn)
                    # compute the federated feature gradients of class c
                    # 计算合成特征的梯度 公式5
                    gw_syn = torch.autograd.grad(loss_syn, net_global_parameters, create_graph=True)
                    # 计算梯度匹配损失 match_loss 并累加到 loss_feature 中 公式6
                    # 为了进一步确保联邦特征和真实特征之间的一致性，使用梯度匹配损失（Zhao，Mopuri和Bilen 2021）来衡量服务器分类器为两类特征生成的梯度之间的差异，并将其最小化
                    loss_feature += match_loss(gw_syn, gw_real_avg[c], args)
            # 计算对比损失 contrast_loss 并累加到 loss_feature 中 公式7
            contrast_loss = self.contras_criterion(self.feature_syn, self.label_syn, new_text_features)
            # Eq. 8
            loss_feature += args.contrast_alpha * contrast_loss
            self.optimizer_feature.zero_grad()
            loss_feature.backward()
            self.optimizer_feature.step()

    def feature_re_train(self, args, fedavg_params):
        feature_syn_train_ft = copy.deepcopy(self.feature_syn.detach())
        label_syn_train_ft = copy.deepcopy(self.label_syn.detach())
        # 创建一个 TensorDataset 对象 dst_train_syn_ft，将合成特征和标签打包成一个数据集
        dst_train_syn_ft = TensorDataset(feature_syn_train_ft, label_syn_train_ft)
        ft_model = nn.Linear(512, args.n_classes).to(self.device)
        optimizer_ft_net = SGD(ft_model.parameters(), lr=args.lr_net)  # optimizer_img for synthetic data
        ft_model.train()
        # 用合成特征训练分类器
        for epoch in range(args.crt_epoch):
            # trainloader_ft = DataLoader(dataset=dst_train_syn_ft,shuffle=True)
            for data_batch in dst_train_syn_ft:
                images, labels = data_batch
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = ft_model(images)
                loss_net = self.criterion(outputs, labels)
                optimizer_ft_net.zero_grad()
                loss_net.backward()
                optimizer_ft_net.step()
        ft_model.eval()
        # 更新全局参数，用线性模型中的对应新参数替换原来的参数。
        feature_net_params = ft_model.state_dict()
        for name_param in reversed(fedavg_params):
            if name_param == 'classifier.bias':
                fedavg_params[name_param] = feature_net_params['bias']
            if name_param == 'classifier.weight':
                fedavg_params[name_param] = feature_net_params['weight']
                break
        return copy.deepcopy(ft_model.state_dict()), copy.deepcopy(fedavg_params)

    def initialize_for_model_fusion(self, list_dicts_local_params: list, list_nums_local_data: list):
        # fedavg
        fedavg_global_params = copy.deepcopy(list_dicts_local_params[0])
        for name_param in list_dicts_local_params[0]:
            list_values_param = []
            for dict_local_params, num_local_data in zip(list_dicts_local_params, list_nums_local_data):
                # 对于每个本地参数字典，将当前参数的值乘以对应的数据量，并将结果添加到 list_values_param 列表中。
                list_values_param.append(dict_local_params[name_param] * num_local_data)
            # 计算加权平均值，得到全局参数的值。
            value_global_param = sum(list_values_param) / sum(list_nums_local_data)
            fedavg_global_params[name_param] = value_global_param
        return fedavg_global_params

    def global_eval(self, fedavg_params, data_test):
        self.syn_model.load_state_dict(fedavg_params)
        self.syn_model.eval()
        with no_grad():
            test_loader = DataLoader(data_test)
            num_corrects = 0
            for data_batch in test_loader:
                images, labels = data_batch
                images, labels = images.to(self.device), labels.to(self.device)
                _, outputs = self.syn_model(images)
                _, predicts = max(outputs, -1)
                num_corrects += sum(eq(predicts.cpu(), labels.cpu())).item()
            accuracy = num_corrects / len(data_test)
        return accuracy

    def download_params(self):
        return self.syn_model.state_dict()

class Local(object):
    def __init__(self,
                 device,
                 data_client,
                 class_list: int,
                 model_dict, args):
        self.data_client = data_client
        self.device = device
        self.class_compose = class_list
        self.criterion = CrossEntropyLoss().to(device)
        self.kd_criterion = KDLoss(T=args.T).to(device)
        self.bkd2_criterion = BKD2Loss(T=args.T).to(device)
        self.local_model = MIL_Attention_fc(**model_dict).to(device)
        self.optimizer = SGD(self.local_model.parameters(), lr=args.lr_local_training)

    # 计算每个类别的真实特征梯度
    def compute_gradient(self, global_params, args):
        # compute C^k
        # 获取类别列表 list_class 和每个类别的样本数量 per_class_compose
        list_class, per_class_compose = get_class_num(self.class_compose)  # class组成

        images_all = []
        labels_all = []
        # 一个字典，用于存储每个类别的样本索引
        indices_class = {class_index: [] for class_index in list_class}
        # 将客户端数据中的图像和标签分别存储到 images_all 和 labels_all 列表中
        for data_batch in self.data_client:
            images, labels = data_batch
            images_all.append(unsqueeze(images, dim=0).resize_(1, *images.shape[1:]))
            labels_all.append(labels)
        #  遍历 labels_all，将每个标签对应的索引添加到 indices_class 字典中。
        for i, lab in enumerate(labels_all):
            indices_class[lab.item()].append(i)
        # images_all = [unsqueeze(self.data_client[i][0], dim=0) for i in range(len(self.data_client))]
        # labels_all = [self.data_client[i][1] for i in range(len(self.data_client))]
        # # 遍历 labels_all，将每个标签对应的索引添加到 indices_class 字典中。
        # for i, lab in enumerate(labels_all):
        #     indices_class[lab].append(i)
        images_all = torch.cat(images_all, dim=0).to(self.device)
        labels_all = torch.tensor(labels_all, dtype=torch.long, device=self.device)
        # 获取指定类别的图像：从指定类别 c 中随机获取 n 张图像。
        def get_images(c, n):  # get random n images from class c
            idx_shuffle = np.random.permutation(indices_class[c])[:n]
            return images_all[idx_shuffle]
        
        # 将全局模型参数加载到本地模型中
        self.local_model.load_state_dict(global_params)
        # 将本地模型设置为评估模式，但将分类器部分设置为训练模式。
        self.local_model.eval()
        self.local_model.classifier.train()
        # 获取分类器的参数列表
        net_parameters = list(self.local_model.classifier.parameters())
        criterion = CrossEntropyLoss().to(self.device)
        # gradients of all classes 存储所有类别的梯度和平均梯度
        truth_gradient_all = {index: [] for index in list_class}
        truth_gradient_avg = {index: [] for index in list_class}
        
        # choose to repeat 10 times
        # 重复10次计算，每次遍历所有类别，从每个类别中获取 args.batch_real 张图像。
        for num_compute in range(10):
            for c, num in zip(list_class, per_class_compose):
                img_real = get_images(c, args.batch_real)
                # print("img_real: ", img_real.shape)
                # transform
                # # 如果启用了数据增强（DSA），则对图像进行数据增强
                # if args.dsa:
                #     seed = int(time.time() * 1000) % 100000
                #     img_real = DiffAugment(img_real, args.dsa_strategy, seed=seed, param=args.dsa_param)

                lab_real = torch.ones((img_real.shape[0],), device=self.device, dtype=torch.long) * c
                feature_real, output_real = self.local_model(img_real)
                loss_real = criterion(output_real, lab_real)
                # compute the real feature gradients of class c
                gw_real = torch.autograd.grad(loss_real, net_parameters)
                gw_real = list((_.detach().clone() for _ in gw_real))
                truth_gradient_all[c].append(gw_real)


        for i in list_class:
            gw_real_temp = []
            gradient_all = truth_gradient_all[i]
            weight = 1.0 / len(gradient_all)
            for name_param in range(len(gradient_all[0])):

                list_values_param = []
                for client_one in gradient_all:
                    list_values_param.append(client_one[name_param] * weight)
                value_global_param = sum(list_values_param)
                gw_real_temp.append(value_global_param)
            # the real feature gradients of all classes
            truth_gradient_avg[i] = gw_real_temp
        return truth_gradient_avg

    def local_train(self, args, global_params, clip_model, text_features):
        self.local_model.load_state_dict(global_params)
        self.local_model.train()
        for data_batch in self.data_client:
            # images, labels, clip_images = data_batch
            images, labels = data_batch
            print("images1: ", images.shape)
            # Apply CLIP model's image transformations
            transform = preprocess
            clip_images = images.clone()
            # Reshape images to (N, C, H, W)
            clip_images = clip_images.view(-1, 1, 32, 32)  # Assuming the original images are 32x32 grayscale images
            print("images2: ", clip_images.shape)
            clip_images = clip_images.unsqueeze(0)  # Add batch dimension
            clip_images = clip_images.repeat(1, 3, 1, 1)  # Convert to 3-channel images
            clip_images = clip_images.squeeze(0)  # Remove batch dimension
            from torchvision.transforms.functional import to_pil_image
            clip_images = transform(to_pil_image(clip_images[0]))

            # from PIL import Image
            # clip_images = transform(Image.fromarray(images.squeeze().cpu().numpy().astype('uint8')))
            images, labels, clip_images = images.to(self.device), labels.to(self.device), clip_images.to(self.device)  # tensor
            print("images3: ", clip_images.shape)
            # compute client's output
            _, outputs = self.local_model(images)
            outputs = outputs.float()

            # get clip feature encode
            with torch.no_grad():
                # 使用 CLIP 模型 clip_model 对 clip_images 进行编码，得到图像特征 image_features
                image_features = clip_model.encode_image(clip_images)
                # clip_images = clip_images.unsqueeze(0)  # Add batch dimension
                # clip_images = torch.nn.functional.interpolate(clip_images, size=(224, 224))
                # clip_images = clip_images.squeeze(0)  # Remove batch dimension
                # image_features = clip_model.encode_image(clip_images)
            image_features = image_features.float()
            image_features /= image_features.norm(dim=-1, keepdim=True)
            # 图像特征与文本特征 text_features 的点积，并乘以一个缩放因子 100.0。
            clip_logits = (100.0 * image_features @ text_features.T)
            #Eq. 1
            # 计算交叉熵损失 loss1，用于衡量模型输出 outputs 与真实标签 labels 之间的差异。
            loss1 = self.criterion(outputs, labels)
            # 计算知识蒸馏损失 loss2，用于衡量模型输出 outputs 与 CLIP 生成的 clip_logits 之间的差异
            loss2 = self.kd_criterion(outputs, clip_logits)
            # 将两种损失加权求和，得到总损失 loss。其中，args.alpha 是一个权重参数，用于平衡两种损失的贡献。
            # 公式1
            loss = loss1 + args.alpha * loss2
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
        return self.local_model.state_dict()

def get_class_num(class_list):
    index = []
    compose = []
    for class_index, j in enumerate(class_list):
        if j != 0:
            index.append(class_index)
            compose.append(j)
    return index, compose
