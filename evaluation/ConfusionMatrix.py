# ------------------------------------------------------------------------------
# @author = 降维虫子
# E-mail  = garfield@outlook.com
# 2022/6/1 20:25
# ------------------------------------------------------------------------------
"""
该脚本用于计算两张二维图片的混淆矩阵
在Pytorch中，输出的预测batch和标签label的size如下
            -label.shape     = torch.Size([2, 256, 256])
            -outputs.shape   = torch.Size([2, 2, 256, 256])
            -dtype           = torch.tensor()
            -location        = CPU/GPU(Cuda)
            ref1: https://blog.csdn.net/sinat_29047129/article/details/103642140
            ref2: https://blog.csdn.net/qq_38253797/article/details/123991536
"""

"""
二分类混淆矩阵
|
|------------------------------------------
|                      预测                 
|------------------------------------------
|                   P       N    
|------------------------------------------
|    真    | P |   TP   |   FN   |  
|    实    | N |   FP   |   TN   | 
|------------------------------------------
|P:Positive
|N:Negative
|------------------------------------------


多类别混淆矩阵
|
|------------------------------------------
|                        预测值             
|------------------------------------------
|                   类别1    类别2     类别3
|------------------------------------------
|    真    | 类别1 |   1   |   2   |   3   |
|    实    | 类别2 |   1   |   1   |   1   |
|    值    | 类别3 |   3   |   1   |   6   |
|-------------------------------------------
|
某个样本标签是类别1（对应真实值），实际分类结果是类别*（对应预测值），则 (真实值类别, 预测值类) 处 +1

口诀：对角全为对，横看是真实，竖看是预测
解释：混淆矩阵对角元素全是预测正确的，数字的值表示各类别预测正确的数目；
     横（行）的数字求和，表示某类别真实值的个数，竖（列）的数字求和，表示模型预测为该类别的个数！
     
准确率（Accuracy）--语义分割的像素准确率 PA
公式：Accuracy = (TP + TN) / (TP + TN + FP + FN)
意义：对角线计算。预测结果中正确的占总预测值的比例（对角线元素值的和 / 总元素值的和）

精准率（Precision）--语义分割的类别像素准确率 CPA、精准率即查准率
公式：Precision = TP / (TP + FP) 或 TN / (TN + FN)
意义：竖着计算(对角值/(对角对应列之和))。预测结果中，某类别预测正确的概率 (numpy的二维矩阵中axis=0 则是按照每列求和, 返回行size)

召回率（Recall），不对应语义分割常用指标, 即查全率
公式：Recall = TP / (TP + FN) 或 TN / (TN + FP)
意义：横着计算(对角值/(对角对应行之和))。真实值中，某类别被预测正确的概率  (numpy的二维矩阵中axis=1 则是按照每行求和, 返回列size)

"""

import numpy as np

# 计算混淆矩阵
# label = np.array([[0, 1, 0], [2, 1, 0], [2, 2, 1]])
# pred = np.array([[1, 1, 0], [2, 1, 1], [2, 2, 1]])
def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(n_class * label_true[mask].astype(int) + label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist


class SegmentationMetric(object):
    # label = np.array([[0, 1, 0], [2, 1, 0], [2, 2, 1]])
    # pred = np.array([[1, 1, 0], [2, 1, 1], [2, 2, 1]])
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def calConfusionMatrix(self, predict, label):
        # 需要统一格式
        if len(predict.shape) == 4:
            predict = predict.argmax(dim=1)
        if not isinstance(predict, np.ndarray):
            if label.device.type == "cuda":
                label = label.cpu().detach().numpy().astype(np.int64)
            else:
                label = label.numpy().astype(np.int64)
            if predict.device.type == "cuda":
                predict = predict.cpu().detach().numpy().astype(np.int64)
            else:
                predict = predict.numpy().astype(np.int64)
        # print("predict.shape, label.shape:", predict.shape, label.shape)
        mask = (label >= 0) & (label < self.num_classes)  # 寻找target中为目标的像素索引
        new_label = self.num_classes * label[mask] + predict[mask]  # 二维的predict拉到一维
        count = np.bincount(new_label, minlength=self.num_classes ** 2)
        confusionMatrix_Step = count.reshape(self.num_classes, self.num_classes)
        return confusionMatrix_Step

    def recall(self, confusionMatrix_sum):
        """
        查全率,行之和, axis = 1
        :return: [Micro-recall , r1, r2, ..., rn] -> [宏, 类别1, ..., 类别n]
        """
        Recall = np.diag(confusionMatrix_sum) / confusionMatrix_sum.sum(axis=1)
        return [np.nanmean(Recall)] + Recall.tolist()

    def precision(self, confusionMatrix_sum):
        """
        查准率,列之和, axis = 0
        :return: [Macro-precision , p1, p2, ..., pn] -> [宏, 类别1, ..., 类别n]
        """
        Precision = np.diag(confusionMatrix_sum) / confusionMatrix_sum.sum(axis=0)
        return [np.nanmean(Precision)] + Precision.tolist()

    def f1(self, confusionMatrix_sum):
        r = np.nanmean(np.diag(confusionMatrix_sum) / confusionMatrix_sum.sum(axis=1))
        p = np.nanmean(np.diag(confusionMatrix_sum) / confusionMatrix_sum.sum(axis=0))
        f1 = 2 * p * r / (p + r)
        return f1

    def meanIOU(self, confusionMatrix_sum):
        intersection = np.diag(confusionMatrix_sum)  # 对角线元素
        union = np.sum(confusionMatrix_sum, axis=1) + np.sum(confusionMatrix_sum, axis=0) - np.diag(confusionMatrix_sum)
        IOU = intersection / union
        mIOU = np.nanmean(IOU)
        return [mIOU] + IOU.tolist()


if __name__ == '__main__':
    import torch
    testType = "tensor"

    if testType == 'numpy':
        Predict = np.array([[0, 0, 1, 0, 2, 2],
                            [0, 1, 1, 1, 2, 2]])  # 可直接换成预测图片
        Label = np.array([[0, 0, 1, 1, 2, 2],
                          [0, 0, 1, 1, 2, 2]])  # 可直接换成标注图片
    if testType == 'tensor':
        Predict = torch.Tensor([[[0, 0, 1, 0, 2, 2],
                                 [0, 1, 1, 1, 2, 2]],
                                [[0, 0, 1, 0, 2, 2],
                                 [0, 1, 1, 1, 2, 2]]
                                ]).long()  # 可直接换成预测图片
        Label = torch.Tensor([[[1, 0, 1, 1, 2, 2],
                               [0, 0, 1, 1, 2, 2]],
                              [[0, 0, 1, 1, 2, 2],
                               [0, 0, 1, 1, 2, 2]]
                              ]).long()  # 可直接换成标注图片
        print("Predict.size", Predict.shape)
        print("Label.size", Predict.shape)

    metric = SegmentationMetric(3)  # 实例化, 3表示有3个分类，有几个分类就填几
    confusionMatrix = np.zeros((3, 3)).astype(int)  # 训练用的混淆矩阵
    for _ in range(3):
        confusionMatrix += metric.calConfusionMatrix(Predict, Label)
    IOU = metric.meanIOU(confusionMatrix)
    f1 = metric.f1(confusionMatrix)
    recall = metric.recall(confusionMatrix)
    precision = metric.precision(confusionMatrix)
    print(confusionMatrix)
    print('IoU is : \n', IOU, "\n")
    print('recall:\n', recall, "\n")
    print('precision:\n', precision, "\n")
    print('f1:\n', f1)
"""
[[5 2 0]
 [3 6 0]
 [0 0 8]]
IoU is : 
 [0.6818181818181818, 0.5, 0.5454545454545454, 1.0] 

recall:
 [0.7936507936507936, 0.7142857142857143, 0.6666666666666666, 1.0] 

precision:
 [0.7916666666666666, 0.625, 0.75, 1.0] 

f1:
 0.7926574885273258

Process finished with exit code 0

"""