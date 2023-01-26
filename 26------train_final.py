#!/usr/bin/env python
# coding: utf-8

#===================================== 导入所需要用到的包 ============================================
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision
import time
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image               
import os
from tqdm import tqdm, trange              #画进度条
import glob                          #获取文件夹中文件的列表
import argparse                       #从命令行加载参数
from torch.utils.data import DataLoader      #用来加载数据集
from prettytable import PrettyTable         #用来画表格，来显示训练过程中的
import json                         #用来存json文件

#指标
import metric                       #分割的评价指标
from measure import SegmentationMetric      #分割的评价指标
from dataset import img_dataset           #加载数据集
from early_stopping import EarlyStopping     #提前停止，保存模型

#model
from model_file.MACUNet import MACUNet      #MACUNet模型
from model_file.MANet import MANet         #MACUNet模型
from model_file.DRCANet import DRCANet      #DRCANet模型
from model_file.UNet import UNet          #UNet模型   
from model_file.ICNet.model import ICNet     #ICNet模型 
from model_file.BiSeNet.model import BiSeNet  #BiseNet模型 双边网络

import warnings                      #忽略警告        
warnings.simplefilter(action='ignore', category=FutureWarning) 
torch.backends.cudnn.enabled =True


#==================================================== 从命令行加载参数 ===================================================
def parse_args():
    parser = argparse.ArgumentParser(description="RemoteSensingSegmentation by PyTorch")
    # dataset
    parser.add_argument('--dataset_name', type=str, default='Vaihingen')     #数据集的名字
    parser.add_argument('--train_data_root', type=str, default='/root/master/ATL/dataset/GID_512/Train')  #训练集的根目录
    parser.add_argument('--val_data_root', type=str, default='/root/master/ATL/dataset/GID_512/Val')    #验证集的根目录
    parser.add_argument('--train_batch_size', type=int, default=4, metavar='N', #训练集的batch_size
                        help='batch size for training (default:2)')  
    parser.add_argument('--val_batch_size', type=int, default=1, metavar='N',  #验证集的batch_size 
                        help='batch size for Val (default:1)')    
    parser.add_argument('--experiment-start-time', type=str,             #开始的时间 05-08-15_57_36
                        default=time.strftime('%m-%d-%H:%M:%S', time.localtime(time.time())).replace(':', '_')) 
    
    # model
    parser.add_argument('--model', type=str, default='macunet', help='model name')   #需要用的模型
    #parser.add_argument('--pretrained', action='store_true', default=True) #action=‘store_true’，只要运行时该变量有传参就将该变量设为True。
    
    # learning_rate
    parser.add_argument('--base_lr', type=float, default=0.0003, metavar='M', help='') #学习率，0.0003

    # environment
    parser.add_argument('--use_cuda', action='store_true', default=True, help='using CUDA training')  #使用cuda
    parser.add_argument('--num_GPUs', type=int, default=1, help='numbers of GPUs')              #GPU的数量，默认为1
    parser.add_argument('--num_workers', type=int, default=12)                           #num_workers的数量，默认为12
    parser.add_argument('--main_device',action='store_true',default=False) #要更换主卡的时候，需要调用这个
    parser.add_argument('--gpu_rank', type=str, default=None)  #gpu的排序，第一个为主卡 
    
    # validation
    parser.add_argument('--total_epochs', type=int, default=100, metavar='N',       #总共训练的轮数
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--only_val',action='store_true',default=False,help='only val') #只验证？只验证的话，会加载下面的模型权重，只运行trainer.Val()
    parser.add_argument('--weights_path',type=str,                         #要加载的模型的权重
               default='/root/master/ATL/code/MACUNet/checkpoint/Vaihingen/macunet/05-05-14_20_29/model_weights.pth',help='weights_path')
    
    
    
    #解析参数！！！
    args = parser.parse_args()   
    
    #创建存对应模型的一个文件夹 
    model_save_path = os.path.join('./checkpoint',args.dataset_name,args.model,args.experiment_start_time)#保存的模型的参数
    args.model_save_path = model_save_path
    
    if not os.path.exists(model_save_path) : #不存在且不是单验证的时候 创建目录
        os.makedirs(model_save_path)
    
    config_file = os.path.join(model_save_path, 'config.json')  #会在这个下面生成一个config.json文件,然后把args写进去
    with open(config_file, 'w') as file:
        json.dump(vars(args), file, indent=4) #漂亮的输出参数

    if args.use_cuda:
        print(f'Train model: {args.model} on {args.num_GPUs} GPU , GPU rank {args.gpu_rank} ') #打印用的gpu的数量
    else:
        print("Using CPU")
    return args

#======================================================Trainer==================================================================
class Trainer(object):  #训练的函数
    def __init__(self, args):
        self.args = args #把参数搞回来
        
        #================================  通过数据集的名字去判断类别数  =====================
        if args.dataset_name == 'GID' or 'Vaihingen'or 'Potsdam':
            self.num_classes = 6
            
        elif args.dataset_name == 'LoveDA':
            self.num_classes = 8
            
        #================================  通过数据集的名字去选择类别的名字  =====================
        
        if args.dataset_name == 'LoveDA':
            self.class_names = ['IGNORE','Background','Building','Road','Water','Barren','Forest','Agricultural'] #LoveDA 8类算上忽略的  
            
        elif args.dataset_name == 'GID': 
            self.class_names = ['Background','Building','Farmland','Forest','Meadow','Water']#GID  6类  

        elif args.dataset_name == 'Vaihingen' or 'Potsdam':
            self.class_names = ['Clutter','Building','Low veg','Tree','Car','Imp.Surf.'] #Vaihingen 6类
            
        #================================ 对数据集进行的transform ==========================
        train_img_transform = torchvision.transforms.Compose([    #对图像做的
                                    torchvision.transforms.ToTensor(),#先除以255归一化
                                    torchvision.transforms.Normalize(mean=[.485, .456, .406],std=[.229, .224, .225])
                                    ])
    
        val_img_transform = torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),#先除以255归一化
                                    torchvision.transforms.Normalize(mean=[.485, .456, .406],std=[.229, .224, .225]
                                    )])
        
        #============================== 通过数据集的名字，去找对应的路径 ===================
        
        if args.dataset_name == 'GID':
            args.train_data_root = '/root/master/ATL/dataset/GID_512/Train'
            args.val_data_root = '/root/master/ATL/dataset/GID_512/Val'
        
        elif args.dataset_name == 'Vaihingen':
            args.train_data_root = '/root/master/ATL/dataset/Vaihingen/Vaihingen_512/Train'
            args.val_data_root = '/root/master/ATL/dataset/Vaihingen/Vaihingen_512/Val'            
        
        elif args.dataset_name == 'LoveDA':
            args.train_data_root = '/root/master/ATL/dataset/LoveDA_512/Train'
            args.val_data_root = '/root/master/ATL/dataset/LoveDA_1024/Val/Urban'        
        
        elif args.dataset_name == 'Potsdam':
            args.train_data_root = '/root/master/ATL/dataset/Potsdam_512/Train'
            args.val_data_root = '/root/master/ATL/dataset/Potsdam_512/Val'            
        
        

        self.train_dataset = img_dataset(args.train_data_root,train_img_transform)
        self.train_loader = DataLoader(dataset=self.train_dataset,
                                       batch_size=args.train_batch_size * args.num_GPUs,  # 这里注意batch size要对应放大倍数
                                       num_workers=args.num_workers,
                                       shuffle=True,
                                       drop_last=True)
        
        self.val_dataset = img_dataset(args.val_data_root,val_img_transform)
        self.val_loader = DataLoader(dataset= self.val_dataset,
                                     batch_size = args.val_batch_size , 
                                     num_workers=args.num_workers,
                                     shuffle = True)
        
        #打印训练集加载信息
        print('-----------------------------------------')    
        if self.train_dataset.__len__()!= 0:
            print(f'  -Train Dataset load successfully! total {self.train_dataset.__len__()}')
            print('  -Train Dataset images shape',self.train_dataset.__getitem__(0)[0].shape)
            print('  -Train Dataset label shape',self.train_dataset.__getitem__(0)[1].shape)
        else:
            print('  --Train dataset load fail--')
        #打印验证集加载信息
        print('-----------------------------------------')    
        if self.val_dataset.__len__()!= 0:
            print(f'  -Val Dataset load successfully! total {self.val_dataset.__len__()}')
            print('  -Val Dataset images shape',self.val_dataset.__getitem__(0)[0].shape)
            print('  -Val Dataset label shape',self.val_dataset.__getitem__(0)[1].shape)
        else:
            print('  --Val dataset load fail--')
        print('-----------------------------------------')    
        
        print(f'class names {self.class_names}.')  #打印数据集的类别名
        
        
        
        #===============================  model  ===================================
        if args.model == 'spnet':  
            self.model = SPNet(block=SE_BasicBlock, layers=[2, 2, 2, 2], num_group=1, n_classes=self.num_classes)
            state_dict = torch.load(
                    './checkpoint/Vaihingen/spnet/none/11-14-18_03_12/epoch_97_acc_0.85741_kappa_0.80884_mIoU_0.56513.pth')
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]
                new_state_dict[name] = v
            self.model.load_state_dict(new_state_dict)
        
        elif args.model == 'enet':
            self.model = ENet(num_classes=self.num_classes)
            
        elif args.model == 'erfnet':
            self.model = ERFNet(num_classes=self.num_classes)

        elif args.model == 'icnet':
            self.model = ICNet(nclass=self.num_classes)

        elif args.model == 'bisenet':
            self.model = BiSeNet(num_classes=self.num_classes, context_path='resnet18')

        elif args.model == 'drcanet':
            self.model = DRCANet(n_classes=self.num_classes)

        elif args.model == 'emanet':
            self.model = EMANet(n_classes=self.num_classes)

        elif args.model == 'dfanet':
            self.model = DFANet(n_classes=self.num_classes)
        
        elif args.model =='macunet':
            self.model = MACUNet(3,self.num_classes)
        
        elif args.model =='manet':
            self.model = MANet(3,self.num_classes)
        
        elif args.model == 'unet':
            self.model = UNet(3,self.num_classes)
            
        
        #============================根据gpu数量，调整是否多gpu训练==========================
        os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
        
        if args.main_device: #主设备不是0的话，
            os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_rank         

#         else:
#             self.device_ids = list(range(torch.cuda.device_count()))  #所有可用的gpu列表 [0,1]

        if args.use_cuda:      
             self.model.cuda()
        if args.num_GPUs > 0:
             self.model = torch.nn.DataParallel(self.model) # 声明所有可用设备
        
        self.model = self.model.cuda()  # 模型放在主设备
        
        #================================= 保存模型的地方 ======================================
        
        self.model_save_path = args.model_save_path #保存模型的地方
        
        #=================================== loss &  optimizer =======================================
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)   #交叉熵为损失  #把第0类忽略不计，因为LoveDA 0类忽略，Vaihingen 0类也忽略，杂乱的
        #self.criterion1 = SoftIoULoss(n_classes=self.num_classes)
    
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=args.base_lr)
        self.metric = SegmentationMetric(self.num_classes)      #评价标准？
        self.early_stopping = EarlyStopping(patience=10, verbose=True) #提前停止，连续多少轮不降，就停止？100吧   
        

    def training(self):
        start = time.time()
        self.model.train()
        lr_adjust = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, args.total_epochs, eta_min=args.base_lr*0.001, last_epoch=-1)#learning_rate * 0.01
        
        conf_mat = np.zeros((self.num_classes, self.num_classes)).astype(np.int64)##########
        
        for epoch in range(1, args.total_epochs + 1):
            tbar = tqdm(self.train_loader) #画个进度条
            for _, image in enumerate(tbar): #获取一张图和 一个label ？
                initial_image = image[0]
                semantic_image = image[1]
                
                initial_image = initial_image.cuda()   #放到GPU上 device=self.device_ids[args.main_device]
                semantic_image = semantic_image.cuda()  #放到GPU上 device=self.device_ids[args.main_device]
                semantic_image_pred = self.model(initial_image) #网络预测值

                loss = self.criterion(semantic_image_pred, semantic_image.long())#算loss .long()
                # print(loss)
                self.optimizer.zero_grad()#梯度设为0
                loss.backward()  #反向传播
                self.optimizer.step() #更新参数
                tbar.set_description(f'epoch {epoch},  training loss {loss}.') #with learning rate {lr_adjust.get_last_lr()[0]}
                
                # delete caches
                del initial_image, semantic_image, loss, semantic_image_pred
                torch.cuda.empty_cache()

            lr_adjust.step() #每一个epoch，学习率调整
            
             
            with torch.no_grad(): #这里用验证集去算mIoU啥的
                self.model.eval()
                for initial_image, semantic_image in tqdm(self.val_loader, desc='val'):

                    initial_image = initial_image.cuda()   #放到GPU上 device=self.device_ids[args.main_device]
                    semantic_image = semantic_image.cuda() #放到GPU上 device=self.device_ids[args.main_device]

                    semantic_image_pred = self.model(initial_image).detach() #detach()就不会传播梯度了，grad属性没有值
                    semantic_image_pred = F.softmax(semantic_image_pred.squeeze(), dim=0)
                    semantic_image_pred = semantic_image_pred.argmax(dim=0)#通道维去argmax，返回的就是对应的类别了
                    
                    #为了计算指标,需要放到cpu上
                    semantic_image = torch.squeeze(semantic_image.cpu(), 0)
                    semantic_image_pred = torch.squeeze(semantic_image_pred.cpu(), 0)

                    self.metric.addBatch(semantic_image_pred, semantic_image)

                    preds = semantic_image_pred.data.cpu().numpy().astype(np.uint8)
                    masks = semantic_image.data.cpu().numpy().astype(np.uint8)

                    conf_mat += metric.confusion_matrix(pred=preds.flatten(),
                                            label=masks.flatten(),
                                            num_classes=self.num_classes)
                                    # delete caches
                    del initial_image, semantic_image, semantic_image_pred
                    torch.cuda.empty_cache()
            
            val_acc, val_acc_per_class, val_F1, val_IoU, val_mean_IoU, val_kappa, val_F1_per_class = metric.evaluate(conf_mat)

            table = PrettyTable(["Num", "Class_name", "acc", "IoU", "F1"])
            for i in range(self.num_classes):
                table.add_row([i, self.class_names[i], val_acc_per_class[i], val_IoU[i], val_F1_per_class[i]])
            print(table)
            print("val_acc:", val_acc)
            print("val_mean_IoU:", val_mean_IoU)
            print("kappa:", val_kappa)
            print('val_F1:', val_F1)

            mIoU = self.metric.meanIntersectionOverUnion() #计算mIoU
            print('mIoU: ', mIoU)
            self.metric.reset()
            self.model.train()

            self.early_stopping(1 - mIoU, self.model, '%s/' % self.model_save_path+'model_weights.pth') #保存训练好的模型
            
            if self.early_stopping.early_stop:
                break
            print('')#打印一行空的，为了分割一轮和一轮，不然分不清
        end = time.time()
        print('Program processed ', end - start, 's, ', (end - start)/60, 'min, ', (end - start)/3600, 'h')
        
        
    def validating(self):
        start = time.time()
        conf_mat = np.zeros((self.num_classes, self.num_classes)).astype(np.int64)
        if args.only_val: #如果只是验证的话，加载程序中的模型进去
            self.model.load_state_dict(torch.load('%s' % args.weights_path))
        
        else:
            if os.path.exists('%s/' % self.model_save_path):
                self.model.load_state_dict(torch.load(self.model_save_path+'/model_weights.pth'))

        self.model.eval()
        for initial_image, semantic_image in tqdm(self.val_loader, desc='val'):

            initial_image = initial_image.cuda()    #放到GPU上device=self.device_ids[args.main_device]
            semantic_image = semantic_image.cuda()  #放到GPU上device=self.device_ids[args.main_device]
            semantic_image_pred = self.model(initial_image).detach() #detach()就不会传播梯度了，grad属性没有值

            semantic_image_pred = F.softmax(semantic_image_pred.squeeze(), dim=0)
            semantic_image_pred = semantic_image_pred.argmax(dim=0)
            
            #为了计算指标,需要放到cpu上
            semantic_image = torch.squeeze(semantic_image.cpu(), 0)
            semantic_image_pred = torch.squeeze(semantic_image_pred.cpu(), 0)

            self.metric.addBatch(semantic_image_pred, semantic_image)
            image = semantic_image_pred
            
            preds = semantic_image_pred.data.cpu().numpy().astype(np.uint8)
            masks = semantic_image.data.cpu().numpy().astype(np.uint8)
            
            conf_mat += metric.confusion_matrix(pred=preds.flatten(),
                                        label=masks.flatten(),
                                        num_classes=self.num_classes)
            
        #=============================画出每一种类别的参数表格==========================
        val_acc, val_acc_per_class, val_F1, val_IoU, val_mean_IoU, val_kappa, val_F1_per_class = metric.evaluate(conf_mat)
        
        metric_file = os.path.join(self.model_save_path, 'best_model_metric.json') 
        
        table = PrettyTable(["Num", "Class_name", "acc", "IoU", "F1"])
        
        with open(metric_file, 'a+') as file:
            json.dump(["Num", "Class_name", "acc", "IoU", "F1"], file)
        
        for i in range(self.num_classes):
            table.add_row([i, self.class_names[i], val_acc_per_class[i], val_IoU[i], val_F1_per_class[i]])
            with open(metric_file, 'a+') as file:
                json.dump([i, self.class_names[i], val_acc_per_class[i], val_IoU[i], val_F1_per_class[i]], file) #漂亮的输出参数

        print(table)
        print("val_acc:", val_acc)
        print("val_mean_IoU:", val_mean_IoU)
        print("kappa:", val_kappa)
        print('val_F1:', val_F1)
        
        out_log = dict(Val_acc=val_acc, 
                   Val_mean_IoU=val_mean_IoU,
                   Kappa=val_kappa,
                   Val_F1=val_F1)
        
        with open(metric_file, 'a+') as file:
                json.dump(out_log, file, indent=4) #漂亮的输出参
        
        

        end = time.time()
        print('Program processed ', end - start, 's, ', (end - start)/60, 'min, ', (end - start)/3600, 'h')
        mIoU = self.metric.meanIntersectionOverUnion()
        print('Val_mIoU: ', mIoU)

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    args = parse_args()
#     writer = SummaryWriter(args.directory)
    trainer = Trainer(args)
    if args.only_val:
        trainer.validating()
    else:
        trainer.training()
        trainer.validating()





