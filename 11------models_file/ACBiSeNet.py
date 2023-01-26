import torch
from torchvision import models
from torch import nn
import warnings
warnings.filterwarnings(action='ignore')


class ACBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(ACBlock, self).__init__()
        self.squre = nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1, stride=2) #square的卷积核
        self.cross_ver = nn.Conv2d(in_planes, out_planes, kernel_size=(1, 3), padding=(0, 1), stride=2)  #水平的卷积核
        self.cross_hor = nn.Conv2d(in_planes, out_planes, kernel_size=(3, 1), padding=(1, 0), stride=2)  #垂直的卷积核
        self.bn = nn.BatchNorm2d(out_planes)  #BN
        self.ReLU = nn.ReLU(True)  #ReLU

    def forward(self, x):
        x1 = self.squre(x)
        x2 = self.cross_ver(x)
        x3 = self.cross_hor(x)
        return self.ReLU(self.bn(x1 + x2 + x3)) 


class resnet18(torch.nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        self.features = models.resnet18(pretrained=pretrained) #resnet18的模型
        self.conv1 = self.features.conv1  #用conv1       3->64,size减半   1024->512  1/2
        self.bn1 = self.features.bn1     #用bn1 
        self.relu = self.features.relu    #用ReLU
        self.maxpool1 = self.features.maxpool  #用maxpool       size减半   512->256 1/4
        self.layer1 = self.features.layer1    #第1层   把 64->64 size不变
        self.layer2 = self.features.layer2    #第2层   把 64->128 size减半  256->128 1/8 
        self.layer3 = self.features.layer3    #第3层   把 128->256 size减半 128->64 1/16
        self.layer4 = self.features.layer4    #第4层   把 256->512 size减半 64->32  1/32  全局池化，512,64->1

    def forward(self, input):
        x = self.conv1(input)
        x = self.relu(self.bn1(x))
        x = self.maxpool1(x)
        feature1 = self.layer1(x)  # 1 / 4
        feature2 = self.layer2(feature1)  # 1 / 8
        feature3 = self.layer3(feature2)  # 1 / 16
        feature4 = self.layer4(feature3)  # 1 / 32
        
        # global average pooling to build tail      
        tail = torch.mean(feature4, 3, keepdim=True)  #（batch,C, H, W）-->(batch,C, H, 1)
        tail = torch.mean(tail, 2, keepdim=True)     #(batch,C, H, 1) -->(batch,C, 1, 1),每个feature的通道不变通道上的feature变成均值了 b*c*1*1
        return feature3, feature4, tail     #返回256，64*64   512，32*32  512，1*1 的特征图 


class resnet101(torch.nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.features = models.resnet101(pretrained=pretrained)
        self.conv1 = self.features.conv1
        self.bn1 = self.features.bn1
        self.relu = self.features.relu
        self.maxpool1 = self.features.maxpool
        self.layer1 = self.features.layer1
        self.layer2 = self.features.layer2
        self.layer3 = self.features.layer3
        self.layer4 = self.features.layer4

    def forward(self, input):
        x = self.conv1(input)
        x = self.relu(self.bn1(x))
        x = self.maxpool1(x)
        feature1 = self.layer1(x)  # 1 / 4
        feature2 = self.layer2(feature1)  # 1 / 8
        feature3 = self.layer3(feature2)  # 1 / 16
        feature4 = self.layer4(feature3)  # 1 / 32
        # global average pooling to build tail
        tail = torch.mean(feature4, 3, keepdim=True)
        tail = torch.mean(tail, 2, keepdim=True)
        return feature3, feature4, tail  #


def build_contextpath(name):
    model = {
           'resnet18': resnet18(pretrained=False),
           #'resnet101': resnet101(pretrained=True)
          }
    return model[name]  #返回的是那个网络




class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2,padding=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, input):
        x = self.conv1(input)
        return self.relu(self.bn(x))

class Spatial_path(torch.nn.Module):     #Spatial path
    def __init__(self):
        super().__init__()
        self.convblock1 = ACBlock(in_channels=3, out_channels=64)    
        self.convblock2 = ACBlock(in_channels=64, out_channels=128)
        self.convblock3 = ACBlock(in_channels=128, out_channels=256)   #尺寸变为原来的1/8 ，通道数3->256
        

    def forward(self, input):
        x = self.convblock1(input)
        x = self.convblock2(x)
        x = self.convblock3(x)
        return x

class AttentionRefinementModule(torch.nn.Module): 
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)  
        self.bn = nn.BatchNorm2d(out_channels) 
        self.sigmoid = nn.Sigmoid()   
        self.in_channels = in_channels      
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1)) 

    def forward(self, input):
        # global average pooling
        x = self.avgpool(input)  #输入进来的 变成batch * C * 1 * 1,,,------256,64*64-->256,1*1
        assert self.in_channels == x.size(1), 'in_channels and out_channels should all be {}'.format(x.size(1))
        x = self.conv(x)     #单纯的改变通道数     256,1*1
        # x = self.sigmoid(self.bn(x))
        x = self.sigmoid(x)   #做sigmoid         256,1*1
        # channels of input and x should be same
        x = torch.mul(input, x) #把原始输入，和这个权重相乘，通道注意力的意味在里面  256,64*64
        return x


class FeatureFusionModule(torch.nn.Module):
    def __init__(self, num_classes, in_channels):   #把SP的一个和CP的三个融合起来
        super().__init__()
        # self.in_channels = input_1.channels + input_2.channels
        # resnet101 3328 = 256(from context path) + 1024(from spatial path) + 2048(from spatial path)
        # resnet18  1024 = 256(from context path) + 256(from spatial path) + 512(from spatial path)
        self.in_channels = in_channels

        self.convblock = ConvBlock(in_channels=self.in_channels, out_channels=num_classes, stride=1) #输入通道,->最后的类别数，尺寸减半
        self.conv1 = nn.Conv2d(num_classes, num_classes, kernel_size=1) #再做一次。
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(num_classes, num_classes, kernel_size=1) #再做一次
        self.sigmoid = nn.Sigmoid()                         #Sigmoid激活
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))      #全局平均池化


    def forward(self, input_1, input_2):
        x = torch.cat((input_1, input_2), dim=1)      #在通道维度上拼接起来他俩
        assert self.in_channels == x.size(1), 'in_channels of ConvBlock should be {}'.format(x.size(1))
        feature = self.convblock(x) #变成类别数的通道  6,128*128
        x = self.avgpool(feature)  #变成6*1*1

        x = self.relu(self.conv1(x))     #6,1*1 ->6,1*1 ->relu 6,1*1
        x = self.sigmoid(self.conv2(x))   #relu 6,1*1 ->6,1*1 ->sigmoid 6,1*1
        x = torch.mul(feature, x)       #6,128*128 * sigmoid 6 1,1  -> 带有权重的6，128*128
        x = torch.add(x, feature)       #6，128*128 + 6,128*128 ->6,128*128
        return x

class BiSeNet(torch.nn.Module):
    def __init__(self, num_classes, context_path):
        super().__init__()
        # build spatial path
        self.saptial_path = Spatial_path()

        # build context path
        self.context_path = build_contextpath(name=context_path)

        # build attention refinement module  for resnet 101
        if context_path == 'resnet101':
            self.attention_refinement_module1 = AttentionRefinementModule(1024, 1024)
            self.attention_refinement_module2 = AttentionRefinementModule(2048, 2048)
            # supervision block
            self.supervision1 = nn.Conv2d(in_channels=1024, out_channels=num_classes, kernel_size=1)
            self.supervision2 = nn.Conv2d(in_channels=2048, out_channels=num_classes, kernel_size=1)
            # build feature fusion module
            self.feature_fusion_module = FeatureFusionModule(num_classes, 3328)

        elif context_path == 'resnet18':
            # build attention refinement module  for resnet 18
            self.attention_refinement_module1 = AttentionRefinementModule(256, 256)
            self.attention_refinement_module2 = AttentionRefinementModule(512, 512)
            # supervision block
            self.supervision1 = nn.Conv2d(in_channels=256, out_channels=num_classes, kernel_size=1)
            self.supervision2 = nn.Conv2d(in_channels=512, out_channels=num_classes, kernel_size=1)
            # build feature fusion module
            self.feature_fusion_module = FeatureFusionModule(num_classes, 1024)
        else:
            print('Error: unspport context_path network \n')

        # build final convolution
        self.conv = nn.Conv2d(in_channels=num_classes, out_channels=num_classes, kernel_size=1)

        self.init_weight()

        self.mul_lr = []
        self.mul_lr.append(self.saptial_path)
        self.mul_lr.append(self.attention_refinement_module1)
        self.mul_lr.append(self.attention_refinement_module2)
        self.mul_lr.append(self.supervision1)
        self.mul_lr.append(self.supervision2)
        self.mul_lr.append(self.feature_fusion_module)
        self.mul_lr.append(self.conv)

    def init_weight(self):                  #权重初始化
        for name, m in self.named_modules():    #name，model  #返回每一个module,内联函数
            if 'context_path' not in name:
                if isinstance(m, nn.Conv2d):  #判断 m 是不是 nn.Conv2d 的一个实例
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                elif isinstance(m, nn.BatchNorm2d):  #判断 m 是不是 nn.BatchNorm2d的一个实例
                    m.eps = 1e-5
                    m.momentum = 0.1
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, input):
        # output of spatial path
        sx = self.saptial_path(input)              #256，128*128

        # output of context path
        cx1, cx2, tail = self.context_path(input)      #256，64*64   512，32*32  512，1*1
        cx1 = self.attention_refinement_module1(cx1)    #带有权重的256,64*64
        cx2 = self.attention_refinement_module2(cx2)    #带有权重的512,32*32
        cx2 = torch.mul(cx2, tail)                #512,32*32 和512 1*1 相乘,广播机制，按元素相乘-> 512,32*32
        
        # upsampling                          #a.size()[-2:]意思是后两位即，feature map的H,W
        cx1 = torch.nn.functional.interpolate(cx1, size=sx.size()[-2:], mode='bilinear') #把cx1 256,64*64->256,128*128
        cx2 = torch.nn.functional.interpolate(cx2, size=sx.size()[-2:], mode='bilinear') #把cx2 512,32*32->512,128*128
        cx = torch.cat((cx1, cx2), dim=1)  #通道维度上拼接cx1，cx2，变成768,128*128----------------------->

        if self.training == True:       #设置.train()，self.training=True设置.eval()，self.training=False
            cx1_sup = self.supervision1(cx1)  #256,64*64 -> 6,64*64
            cx2_sup = self.supervision2(cx2)  #512,32*32 -> 6,32*32
            cx1_sup = torch.nn.functional.interpolate(cx1_sup, size=input.size()[-2:], mode='bilinear') #把cx1_sup 内插到 6,1024*1024
            cx2_sup = torch.nn.functional.interpolate(cx2_sup, size=input.size()[-2:], mode='bilinear') #把cx2_sup 内插到 6,1024*1024

        # output of feature fusion module
        result = self.feature_fusion_module(sx, cx)  #混合 dx 和 cx，256,128*128 和 768,128*128
                                       #1.通道维度上，拼接起来-> 1024,128*128
                                       #2.1024,128*128 -> 6,128*128

        # upsampling
        result = torch.nn.functional.interpolate(result, scale_factor=8, mode='bilinear') #6,128*128 内插8倍，6,1024*1024
        result = self.conv(result) # 6,1024*1024 -> 6,1024*1024 最后的结果了

        if self.training == True:
            return result

        return result


if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    model = BiSeNet(6, 'resnet18')

    img = torch.rand(1, 3, 512, 512)
    from time import time
    
    a = time()
    output = model(img)
    b = time()

    from thop import profile
    from thop import clever_format

    flops, params = profile(model, inputs=(img,))
    flops, params = clever_format([flops, params], "%.3f")
    
    print('time=',b-a)
    print('out_shape',output.shape)
    print("flops=", flops)  # 40.74G
    print("params=", params)  # 23.09M

