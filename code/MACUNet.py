import torch
from torch import nn

##########################################通道注意力################################################
class ChannelAttention(nn.Module):  #in_planes,out_planes,输入的通道数，输出的通道数
    def __init__(self, in_planes, out_planes, ratio=2):
        super(ChannelAttention, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, 1, bias=False)#CAB中的第一次卷积，
        self.avg_pool = nn.AdaptiveAvgPool2d(1)#全局平均池化
        self.max_pool = nn.AdaptiveMaxPool2d(1)#全局最大值池化

        self.fc11 = nn.Conv2d(out_planes, out_planes // ratio, 1, bias=False) #CAB中的第二次卷积
        self.fc12 = nn.Conv2d(out_planes // ratio, out_planes, 1, bias=False) #CAB中的第三次卷积

        self.fc21 = nn.Conv2d(out_planes, out_planes // ratio, 1, bias=False)
        self.fc22 = nn.Conv2d(out_planes // ratio, out_planes, 1, bias=False)
        self.relu1 = nn.ReLU(True)

        self.sigmoid = nn.Sigmoid() #Sigmoid输出

    def forward(self, x):
        x = self.conv(x) #如Xde3，320*H*W,第一次卷完，变成了128*H*W
        avg_out = self.fc12(self.relu1(self.fc11(self.avg_pool(x)))) #通道注意力的Avg输出，128*H*W-->Avg_pool-->第二次卷积-->ReLU-->第三次卷积
        max_out = self.fc22(self.relu1(self.fc21(self.max_pool(x)))) #通道注意力的Max输出，128*H*W-->Max_pool-->第二次卷积-->ReLU-->第三次卷积
        out = avg_out + max_out #将两个部分求和
        del avg_out, max_out #删除变量，释放内存吧？
        return x * self.sigmoid(out) #求和后过sigmoid，并将这个输出的128*1*1的权重，和最开始的第一次卷积后的128*H*W相乘

##################################LeakeyReLU##############################################
def conv3otherRelu(in_planes, out_planes, kernel_size=None, stride=None, padding=None):
    # 3x3 convolution with padding and relu
    if kernel_size is None:
        kernel_size = 3
    assert isinstance(kernel_size, (int, tuple)), 'kernel_size is not in (int, tuple)!'

    if stride is None:
        stride = 1
    assert isinstance(stride, (int, tuple)), 'stride is not in (int, tuple)!'

    if padding is None:
        padding = 1
    assert isinstance(padding, (int, tuple)), 'padding is not in (int, tuple)!'

    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
        # nn.ReLU()  # inplace=True
        nn.LeakyReLU()
    )


####################################ACB，非对称卷积块的实现################################
class ACBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(ACBlock, self).__init__()
        self.squre = nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1, stride=1) #square的卷积核
        self.cross_ver = nn.Conv2d(in_planes, out_planes, kernel_size=(1, 3), padding=(0, 1), stride=1)  #水平的卷积核
        self.cross_hor = nn.Conv2d(in_planes, out_planes, kernel_size=(3, 1), padding=(1, 0), stride=1)  #垂直的卷积核
        self.bn = nn.BatchNorm2d(out_planes)  #BN
        self.ReLU = nn.ReLU(True)  #ReLU

    def forward(self, x):
        x1 = self.squre(x)
        x2 = self.cross_ver(x)
        x3 = self.cross_hor(x)
        return self.ReLU(self.bn(x1 + x2 + x3)) 

################################网络结构，MACU-Net#############################################
class MACUNet(nn.Module):   
    def __init__(self, band_num, class_num):
        super(MACUNet, self).__init__()
        self.band_num = band_num
        self.class_num = class_num
        self.name = 'MACUNet'

        # channels = [32, 64, 128, 256, 512]
        channels = [16, 32, 64, 128, 256, 512]  #Encoder-Decoder的通道数，为这个。
        self.conv1 = nn.Sequential(            #第一个卷积层
            ACBlock(self.band_num, channels[0]),  #(输入的图片的维度，16)，1024*1024*3->1024*1024*16
            ACBlock(channels[0], channels[0])    #(16,16)
        )
        self.conv12 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)), #最大值池化， 1024*1024*16 --> 512*512*16
            ACBlock(channels[0], channels[1])           #512*512*16 --> 512*512*32
        )
        self.conv13 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)), #512*512*32 --> 256*256*32
            ACBlock(channels[1], channels[2]),          #256*256*32 -->256*256*64
        )
        self.conv14 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)), #256*256*64 -->128*128*64
            ACBlock(channels[2], channels[3])           #128*128*64 --> 128*128*128
        )

        self.conv2 = nn.Sequential(                   
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)), #1024*1024*16 -->512*512*16
            ACBlock(channels[0], channels[1]),          #512*512*16 --> 512*512*32
            ACBlock(channels[1], channels[1])           #512*512*32 --> 512*512*32
        )
        self.conv23 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)), #512*512*32 -->256*256*32
            ACBlock(channels[1], channels[2])           #256*256*32 -->256*256*64
        )
        self.conv24 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)), #256*256*64 --> 128*128*64
            ACBlock(channels[2], channels[3])           #128*128*64 --> 128*128*64
        )

        self.conv3 = nn.Sequential(                   
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),  #512*512*32 --> 256*256*32
            ACBlock(channels[1], channels[2]),           #256*256*32 --> 256*256*64
            ACBlock(channels[2], channels[2]),           #256*256*32 --> 256*256*64
            ACBlock(channels[2], channels[2])           #256*256*32 --> 256*256*64
        )
        self.conv34 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)), #256*256*64 --> 128*128*64
            ACBlock(channels[2], channels[3])           #128*128*64 --> 128*128*128
        )

        self.conv4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),  #256*256*64-->128*128*64
            ACBlock(channels[2], channels[3]),           #128*128*64-->128*128*128
            ACBlock(channels[3], channels[3]),           #128*128*64-->128*128*128
            ACBlock(channels[3], channels[3])           #128*128*64-->128*128*128
        )

        self.conv5 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),#128*128*128 --> 64*64*128
            ACBlock(channels[3], channels[4]),          #64*64*128 --> 64*64*256
            ACBlock(channels[4], channels[4]),          #64*64*256 --> 64*64*256
            ACBlock(channels[4], channels[4])           #64*64*256 --> 64*64*256
        )

        self.skblock4 = ChannelAttention(channels[3]*5, channels[3]*2, 16) #通道注意力生成
        self.skblock3 = ChannelAttention(channels[2]*5, channels[2]*2, 16) #通道注意力
        self.skblock2 = ChannelAttention(channels[1]*5, channels[1]*2, 16) #通道注意力
        self.skblock1 = ChannelAttention(channels[0]*5, channels[0]*2, 16) #通道注意力

        self.deconv4 = nn.ConvTranspose2d(channels[4], channels[3], kernel_size=(2, 2), stride=(2, 2)) #64*64*256 --> 128*128*128
        self.deconv43 = nn.ConvTranspose2d(channels[3], channels[2], kernel_size=(2, 2), stride=(2, 2)) #128*128*128-->256*256*64
        self.deconv42 = nn.ConvTranspose2d(channels[2], channels[1], kernel_size=(2, 2), stride=(2, 2))#256*256*64-->512*512*32
        self.deconv41 = nn.ConvTranspose2d(channels[1], channels[0], kernel_size=(2, 2), stride=(2, 2))#512*512*32-->1024*1024*16

        self.conv6 = nn.Sequential(
            ACBlock(channels[4], channels[3]), #256-->128 Xde4
            ACBlock(channels[3], channels[3]),
        )

        self.deconv3 = nn.ConvTranspose2d(channels[3], channels[2], kernel_size=(2, 2), stride=(2, 2))
        self.deconv32 = nn.ConvTranspose2d(channels[2], channels[1], kernel_size=(2, 2), stride=(2, 2))
        self.deconv31 = nn.ConvTranspose2d(channels[1], channels[0], kernel_size=(2, 2), stride=(2, 2))
        self.conv7 = nn.Sequential(
            ACBlock(channels[3], channels[2]),
            ACBlock(channels[2], channels[2]),
        )

        self.deconv2 = nn.ConvTranspose2d(channels[2], channels[1], kernel_size=(2, 2), stride=(2, 2))
        self.deconv21 = nn.ConvTranspose2d(channels[1], channels[0], kernel_size=(2, 2), stride=(2, 2))
        self.conv8 = nn.Sequential(
            ACBlock(channels[2], channels[1]),
            ACBlock(channels[1], channels[1])
        )

        self.deconv1 = nn.ConvTranspose2d(channels[1], channels[0], kernel_size=(2, 2), stride=(2, 2))
        self.conv9 = nn.Sequential(
            ACBlock(channels[1], channels[0]),
            ACBlock(channels[0], channels[0])
        )

        self.conv10 = nn.Conv2d(channels[0], self.class_num, kernel_size=1, stride=1)

    def forward(self, x):
        conv1 = self.conv1(x) #原始图进来，做第一次卷积 1024*1024*3->1024*1024*16 (Xen1)
        conv12 = self.conv12(conv1)   #512*512*16 --> 512*512*32            (Xen1 做了1次ACB卷积)
        conv13 = self.conv13(conv12)  #512*512*32 --> 256*256*32            (Xen1 做了2次ACB卷积)
        conv14 = self.conv14(conv13)  #256*256*64 --> 128*128*128           (Xen1 做了3ACB卷积)

        conv2 = self.conv2(conv1)    #1024*1024*16 --> 512*512*32          (Xen2)
        conv23 = self.conv23(conv2)   #512*512*32  --> 256*256*64          (Xen2 做了1次ACB卷积)
        conv24 = self.conv24(conv23)  #256*256*64 --> 128*128*64          (Xen2 做了2次ACB卷积)

        conv3 = self.conv3(conv2)   #512*512*32 --> 256*256*64            (Xen3)
        conv34 = self.conv34(conv3)  #256*256*64 --> 128*128*128          (Xen3 做了1次ACB卷积)
        
        conv4 = self.conv4(conv3)   #256*256*64 --> 128*128*128           (Xen4)
        
        conv5 = self.conv5(conv4)   #128*128*128--> 64*64*256            (Xen5)(Xde5)
        
        
        #######去求Xde4
        deconv4 = self.deconv4(conv5)   #64*64*256 --> 128*128*128        (Xde5 做了1次转置卷积) ->为了生成Xde4
        deconv43 = self.deconv43(deconv4) #128*128*128-->256*256*64        (Xde5 做了2次转置卷积) ->为了生成Xde3
        deconv42 = self.deconv42(deconv43)#256*256*64-->512*512*32        (Xde5 做了3次转置卷积)  ->为了生成Xde2
        deconv41 = self.deconv41(deconv42) #512*512*32-->1024*1024*16      (Xde5 做了4次转置卷积)  ->为了生成Xde1

        conv6 = torch.cat((deconv4, conv4, conv34, conv24, conv14), 1) #(生成Xde4用的Xde5 Xen4 生成Xde4用的Xen3  生成Xde4用的Xen2 生成Xde4用的Xen1)
        #deconv4 是Xde5(Xen5)做了一次转置卷积
        #conv4 是直连Xen4
        #conv34 是Xen3做了1次maxpooling
        #conv24 是Xen2做了2次maxpoling
        #conv14 是Xen1做了3次maxpooling
        
        conv6 = self.skblock4(conv6) #做自注意力，128*128* 128*5的通道-->128*128*256
        conv6 = self.conv6(conv6)  #128*128*256-->256*256*128         (Xde4)
        del deconv4, conv4, conv34, conv24, conv14, conv5
        #######去求Xde3
        deconv3 = self.deconv3(conv6) #256*256*128 -->512*512*64(Xde4做转置卷积)  (Xde4 做了1次转置卷积)
        deconv32 = self.deconv32(deconv3)#512*512*64-->1024*1024*32          (Xde4 做了2次转置卷积)
        deconv31 = self.deconv31(deconv32)#1024*1024*32-->2048*2048*16        (Xde4 做了3次转置卷积)

        conv7 = torch.cat((deconv3, deconv43, conv3, conv23, conv13), 1)#(生成Xde3用的Xde4 生成Xde3用的Xde5 Xen3 生成Xde3用的Xen2  生成Xde3用的Xen1)
        conv7 = self.skblock3(conv7)
        conv7 = self.conv7(conv7)                           #(Xde4)
        del deconv3, deconv43, conv3, conv23, conv13, conv6
        #######去求Xde2
        deconv2 = self.deconv2(conv7)                        #(Xde3 做了1次转置卷积)
        deconv21 = self.deconv21(deconv2)                     #(Xde3 做了2次转置卷积)

        conv8 = torch.cat((deconv2, deconv42, deconv32, conv2, conv12), 1)
        conv8 = self.skblock2(conv8)
        conv8 = self.conv8(conv8)
        del deconv2, deconv42, deconv32, conv2, conv12, conv7
        #######去求Xde1
        deconv1 = self.deconv1(conv8)#Xde2
        conv9 = torch.cat((deconv1, deconv41, deconv31, deconv21, conv1), 1)
        conv9 = self.skblock1(conv9)
        conv9 = self.conv9(conv9)
        # conv9 = self.seblock(conv9)
        del deconv1, deconv41, deconv31, deconv21, conv1, conv8

        output = self.conv10(conv9)

        return output


if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    model = MACUNet(3,6)

    img = torch.rand(1, 3, 1024, 1024)
    from time import time
    
    model = model.cuda()
    img = img.cuda()
    
    a = time()
    output = model(img)
    b = time()

    from thop import profile
    from thop import clever_format

    flops, params = profile(model, inputs=(img,))
    flops, params = clever_format([flops, params], "%.3f")
    
    print("flops=", flops)  # 42.494G
    print("params=", params)  # 23.356M
    print('out_shape',output.shape)
    print('time=',b-a)
    print("FPS=",60//(b-a))
    
    