import numpy as np
import PIL.Image as Image
from sklearn.metrics import confusion_matrix, cohen_kappa_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

'''
传入的参数为标签、预测图、类别数
所有的格式均从零0按照顺序
'''


def miou(label, predict, class_num):
    # 创建用于存放的矩阵
    classes = np.linspace(0, class_num - 1, class_num).astype(int)
    # label转换成独热编码
    if label.shape[0] != class_num:
        one_hot_label = np.zeros((class_num, label.shape[0], label.shape[1]))
        for n in classes:  # 将其映射到矩阵
            one_hot_label[n, :, :] = label == n
    # predict转换成独热编码
    if predict.shape[0] != class_num:
        one_hot_predict = np.zeros((class_num, predict.shape[0], predict.shape[1]))
        for n in classes:
            one_hot_predict[n, :, :] = predict == n
            # 类别循环计算交并
    log_matrix = np.zeros((class_num, 2))  # 用于保存每张图每类别的交并面积
    for n in range(class_num):
        log_matrix[n, 0] = np.sum(one_hot_label[n, ...].astype(bool) & one_hot_predict[n, ...].astype(bool))  # 计算交集
        log_matrix[n, 1] = np.sum(one_hot_label[n, ...].astype(bool) | one_hot_predict[n, ...].astype(bool))  # 计算并集
    return log_matrix


# MIOU
def miou_M(MATRIX_C):
    iou = np.diag(MATRIX_C) / (MATRIX_C.sum(axis=1) + MATRIX_C.sum(axis=0) - np.diag(MATRIX_C))
    miou = np.nanmean(iou)
    return iou, miou


# 混淆矩阵
def compute_matrix(gt, pred, class_num=2):
    """
    用于计算网络预测的混淆矩阵
    Args:
        gt: Tensor batch x [W, H ]
        pred: Tensor batch x [C, W, H ]
        class_num: 从0到n

    Returns:

    """
    pred = pred.argmax(dim=0)
    if gt.device.type == "cuda":
        gt = gt.cpu().numpy()
    else:
        gt = gt.numpy()

    if pred.device.type == "cuda":
        pred = pred.cpu().numpy()
    else:
        pred = pred.numpy()

    labels = [x for x in range(class_num)]
    # print(gt.shape, pred.shape)
    matrix = confusion_matrix(y_true=np.array(gt).flatten(), y_pred=np.array(pred).flatten(), labels=labels)
    return matrix


def compute_f1(prediction, target):
    """
    :param prediction: 2d array, int,
            estimated targets as returned by a classifier
    :param target: 2d array, int,
            ground truth
    :return f1: float
    """
    prediction.tolist(), target.tolist()
    img, target = np.array(prediction).flatten(), np.array(target).flatten()
    f1 = f1_score(y_true=target, y_pred=img)
    return f1


def compute_kappa(prediction, target):
    """
    :param prediction: 2d array, int,
            estimated targets as returned by a classifier
    :param target: 2d array, int,
            ground truth
    :return:
        kappa: float
    """
    prediction.tolist(), target.tolist()
    img, target = np.array(prediction).flatten(), np.array(target).flatten()
    kappa = cohen_kappa_score(target, img)
    return kappa


def compute_acc(gt, pred):
    # matrix = confusion_matrix(y_true=np.array(gt).flatten(), y_pred=np.array(pred).flatten())
    acc = np.diag(matrix).sum() / matrix.sum()
    return acc


def compute_recall(gt, pred):
    #  返回所有类别的召回率recall
    # matrix = confusion_matrix(y_true=np.array(gt).flatten(), y_pred=np.array(pred).flatten())
    recall = np.diag(matrix) / matrix.sum(axis=0)
    return recall


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import seaborn as sns

    gt = Image.open("D:\Dataset\FUSAR_Map\FUSAR512/2411_lab.png")
    gt = np.array(gt)
    gt[gt == 255] = 1
    pre = np.random.randint(0, 2, (512, 512))

    labels = [0, 1]
    confusion_matrix_1 = confusion_matrix(y_true=np.array(gt).flatten(), y_pred=np.array(gt).flatten(), labels=labels)
    print(confusion_matrix_1.shape)
    plt.matshow(confusion_matrix_1)
    plt.show()
    sns.heatmap(confusion_matrix_1)
    plt.show()
