import numpy as np


def confusion_matrix(pred, label, num_classes):
    mask = (label >= 0) & (label < num_classes)
    conf_mat = np.bincount(num_classes * label[mask].astype(int) + pred[mask], minlength=num_classes**2).reshape(num_classes, num_classes)
    return conf_mat

def evaluate(conf_mat):
    acc = np.diag(conf_mat)[1:].sum() / conf_mat[1:, 1:].sum() #准确度,这里不算第1类   #对于Vaihingen，Potsdam，LoveDA是这个样的
    acc_per_class = np.diag(conf_mat) / conf_mat.sum(axis=1)
    # acc_cls = np.nanmean(acc_per_class)

    recall_per_class = np.diag(conf_mat) / conf_mat.sum(axis=0)

    acc_per_class_transpose = acc_per_class.T

    multi = 2 * (acc_per_class_transpose * recall_per_class)
    add = acc_per_class + recall_per_class

    F1_per_class = (multi / add)

    F1 = (multi / add).mean()

    IoU = np.diag(conf_mat) / (conf_mat.sum(axis=1) + conf_mat.sum(axis=0) - np.diag(conf_mat))
    mean_IoU = np.nanmean(IoU)

    # 求kappa
    pe = np.dot(np.sum(conf_mat, axis=0), np.sum(conf_mat, axis=1)) / (conf_mat.sum()**2)
    kappa = (acc - pe) / (1 - pe)
    
    return acc, acc_per_class, F1, IoU, mean_IoU, kappa, F1_per_class