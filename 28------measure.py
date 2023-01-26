import numpy as np


class SegmentationMetric(object):                      #指标！！！
    def __init__(self, numClass):
        self.numClass = numClass                      #几类
        self.confusionMatrix = np.zeros((self.numClass,) * 2)   #混淆矩阵，类别*类别的一个矩阵

    def meanIntersectionOverUnion(self):          #计算mIoU
        # Intersection = TP       交集
        # Union = TP + FP + FN     并集
        # IoU = TP / (TP + FP + FN)  交并比   某一类的 Iou =  预测为这类&&真的是这类 / 预测为这类，但是为另一类 + 预测为另一类，但是为这一类
        intersection = np.diag(self.confusionMatrix)  #输出混淆矩阵对角线元素，就是预测对的，交集
        union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(self.confusionMatrix) #NB
         #union怎么算：比如第一类，那就是混淆矩阵的第一行的和+第一列的和-第一个元素，因为多算了一遍，
                 #比如第二类，那就是第二行的和+第二列的和-第二个元素
        IoU = intersection / union
        mIoU = np.nanmean(IoU)
        return mIoU

    def genConfusionMatrix(self, imgPredict, imgLabel):
        # remove classes from unlabeled pixels in gt image and predict #移除不进行预测的那类
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)  #mask = label>=0&类别数，比如6类，0-5 #判断是否所有像素的标签都在0-6里512*512[True/False]
        label = self.numClass * imgLabel[mask] + imgPredict[mask]  #妙哇，imglabel[mask]，只会显示mask为True的值，并存在以为数组中,乘类别数，太妙了
        count = np.bincount(label, minlength=self.numClass ** 2)   
        confusionMatrix = count.reshape(self.numClass, self.numClass)
        return confusionMatrix

    def addBatch(self, imgPredict, imgLabel):
        imgPredict = imgPredict.cpu()
        imgLabel = imgLabel.cpu()
        assert imgPredict.shape == imgLabel.shape
        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel)

    def reset(self):
        self.confusionMatrix = np.zeros((self.numClass, self.numClass))



