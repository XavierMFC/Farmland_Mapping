import sys
import os
import math
import yaml
import torch
import numpy as np
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
from tqdm import tqdm
import argparse

from evaluation.ConfusionMatrix import SegmentationMetric
from data import build_data

from model import segformer



SEED = 1
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


def load_yaml(yaml_path):
    try:
        with open(yaml_path, 'rb') as f:
            # type: dict
            yaml_file = yaml.load(f, Loader=yaml.FullLoader)
        f.close()
    except Exception as ExName:
        print(ExName)
    return yaml_file


def get_dataloader(dataset_mode, file_path=(None,None,None), batch_size=(None,None,None)):
    dataloader_train, dataloader_val, dataloader_test = build_data.BuildData(dataset_mode, 
                                                                             file_path=file_path, 
                                                                             batch_size=batch_size,
                                                                             DDP=False).get_dataset()
    return dataloader_train, dataloader_val, dataloader_test



def get_learn_rate_scheduler(optimizer, learnMode, warm_up_epochs, epochs_train):
    if learnMode == 'cosineAnn':
        scheduler = CosineAnnealingLR(optimizer, T_max=5, eta_min=0)
    if learnMode == 'cosineAnnWarm':
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5)
    if learnMode == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5, last_epoch=-1)
    if learnMode == 'ExponentialLR':
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9, last_epoch=-1)

    if learnMode == 'WarmCosinDecay':
        warm_up_with_cosine_lr = lambda epoch: \
                                (epoch / warm_up_epochs if epoch != 0 else 1 / (warm_up_epochs * 3)) if epoch <= warm_up_epochs else \
                                0.5 * (math.cos((epoch - warm_up_epochs) / (epochs_train - warm_up_epochs) * math.pi) + 1)
                                # 0.01 if 0.5 * (math.cos((epoch - warm_up_epochs) / (epochs_train - warm_up_epochs) * math.pi) + 1) < 0.1 else \
                                
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_with_cosine_lr)
    return scheduler


def get_criterion(device, lossname, total_num_Samples=[1, 1], label_smoothing=0.1, ignore_index=10):

    if total_num_Samples:
        weights_norm = torch.FloatTensor([1 - (x / sum(total_num_Samples)) for x in total_num_Samples]).to(device)  # 根据各类别数量计算权重
    else:
        weights_norm = None

    if lossname == 'segformer_3task_v1':
        criterion1 = torch.nn.CrossEntropyLoss(weight=weights_norm, ignore_index=10, label_smoothing=label_smoothing)
        criterion2 = torch.nn.BCELoss()
        criterion3 = torch.nn.L1Loss()
        return criterion1, criterion2, criterion3
        # criterion1 = dice_loss.DiceLoss()
        # criterion2 = torch.nn.BCELoss()
        # criterion3 = torch.nn.L1Loss()
        # return criterion1, criterion2, criterion3

def SaveWeight(SaveIndex, iou, weight_save_path, net):
    """
    SaveIndex: 存储保存的权重的iou
    iou: 当前epoch的iou
    weight_save_path: 权重保存路径
    net: 当前网络
    """
    # 保存权重
    SaveIndex.sort()
    WeightsSavePath1Root = weight_save_path
    WeightsSavePath1 = WeightsSavePath1Root + "_%f.pt" % iou[0]
    if iou[0] > SaveIndex[0]:
        if os.path.exists(WeightsSavePath1Root + "_%f.pt" % SaveIndex[0]):
            os.remove(WeightsSavePath1Root + "_%f.pt" % SaveIndex[0])
        SaveIndex[0] = iou[0]
        torch.save(net.state_dict(), WeightsSavePath1)
        print('当前文件保存路径 = ', WeightsSavePath1)
        print('保存前N的精度 = ', SaveIndex)
    return SaveIndex


def CM2Index(CM, meanWeights, evaluation):
    iou = evaluation.meanIOU(CM)  # 单类iou
    meanIou = [(np.array(meanWeights) * np.array(iou[1:])).sum()] + iou[1:]  # 加权平均iou
    f1 = evaluation.f1(CM)
    recall = evaluation.recall(CM)
    precision = evaluation.precision(CM)
    return iou, meanIou, f1, recall, precision


def train(device, numcls, epochs_train, net, optimizer, criterion, scheduler, SaveIndex_N, weights_path, category_name, data=(None, None, None)):
    """
    device: 使用硬件
    numcls: 类别数
    epochs_train: 训练的轮次
    net: 模型
    optimizer: 优化器
    criterion: 损失函数
    scheduler: 学习率调整
    SaveIndex_N: 保存最好的N个权重
    weights_path: 权重保存位置
    data: 数据集
    """
    time_count = 0
    print('numcls = ', numcls)
    evaluation = SegmentationMetric(numcls)  # 实例化, numcls表示有numcls个分类
    SaveIndex = [0 for _ in range(SaveIndex_N)]  # 创建存放权重精度的列表
    dataloader_train, dataloader_val, dataloader_test = data
    meanWeights = [1 / numcls for _ in range(numcls)]  # 均匀权重

    for epoch in range(epochs_train):
        print('Epoch {}/{}'.format(epoch, epochs_train - 1))
        net.train()  # 训练模式
        epoch_loss = 0  # 用于统计每个epoch的总损失
        ConfusionMatrix_train = np.zeros((numcls, numcls)).astype(int)  # 训练用的混淆矩阵
        dt_size = len(dataloader_train)

        for step, data in enumerate(dataloader_train):  # 遍历数据集，

            x, y = data
            step += 1
            inputs = x.to(device)
            labels = y.long().to(device)


            labels_mask = labels[:, 0, ...].long()
            labels_edge = labels[:, 1, ...].long()
            labels_dist = labels[:, 2, ...].float()
            # print(labels_bound.shape, labels_seg.shape) torch.Size([2, 1024, 1024]) torch.Size([2, 1024, 1024])

            optimizer.zero_grad()  # 梯度清理->前向传播->损失计算->反向传播->参数更新
            out_seg, out_edge, out_dist = net(inputs)  # print(outputs_bound.shape, outputs_seg.shape) torch.Size([2, 1, 1024, 1024]) torch.Size([2, 2, 1024, 1024])

            loss1 = criterion[0](out_seg, labels_mask)
            loss2 = criterion[1](out_edge[:, 0, ...].float(), labels_edge.float())
            loss3 = criterion[2](out_dist[:, 0, ...].float(), labels_dist.float())

            loss = loss1 + loss2 + loss3

            loss.backward()
            optimizer.step()  # 更新参数,所有的优化器Optimizer都实现了step()方法对所有的参数进行更新


            ConfusionMatrix_train = ConfusionMatrix_train + evaluation.calConfusionMatrix(out_seg, labels_mask)  # 计算每个batch的混淆矩阵

            print("epoch:%d/%d steps:%d/%d  train_loss=%0.3f  lr=%f" % (epoch + 1, epochs_train, step, dt_size, loss.item(), optimizer.param_groups[0]['lr']))

            time_count += 1
            epoch_loss += loss.item()

        scheduler.step()
        iou, meanIou, f1, recall, precision = CM2Index(ConfusionMatrix_train, meanWeights, evaluation)

        print("训练集运行结束 epoch %d Train -> [miou, iou1,..., iouN] = " % epoch, meanIou)
        print("训练集运行结束 epoch %d Train -> f1 = " % epoch, f1)
        print("训练集运行结束 epoch %d Train -> recall = " % epoch, recall)
        print("训练集运行结束 epoch %d Train -> precision = " % epoch, precision)

        print("epoch %d loss:%0.3f" % (epoch + 1, epoch_loss))

        # 测试集精度评价
        if (epoch + 1) % 1 == 0:
            """ 
            每隔固定回合测试一次
            除了写入图像、还有数据可视化
            """
            net.eval()
            with torch.no_grad():
                ConfusionMatrix_test = np.zeros((numcls, numcls)).astype(int)
                for index, data in tqdm(enumerate(dataloader_val)):
                    x, labels = data
                    x = x.to(device)
                    labels = labels.to(device).long()

                    labels_mask = labels[:, 0, ...].long()
                    labels_edge = labels[:, 1, ...].long()
                    labels_dist = labels[:, 2, ...].float()

                    out_seg, out_edge, out_dist = net(x)

                    ConfusionMatrix_test = ConfusionMatrix_test + evaluation.calConfusionMatrix(out_seg, labels_mask)  # 计算每个batch的混淆矩阵
                    # Visual_ins.makeVizGrid(x, outputs_seg, index, epoch, p=100)
                
                """
                精度结果
                viz.line([[0.0, 0.0, 0.0, 0.0]], [0.], win='Test/IOU', opts=dict(title='IOU/Test', legend=['IOU/Test', 'F1']))
                viz.line([[0.0, 0.0, 0.0, 0.0]], [0.], win='Test/Recall', opts=dict(title='Recall/Test', legend=['R', 'R_0', 'R_1', 'F1']))
                viz.line([[0.0, 0.0, 0.0, 0.0]], [0.], win='Test/Precision', opts=dict(title='Precision/Test', legend=['P', 'P_0', 'P_1', 'F1']))
                """
                iou, meanIou, f1, recall, precision = CM2Index(CM=ConfusionMatrix_test, meanWeights=meanWeights, evaluation=evaluation)

                print("测试集运行结束--epoch %d Test -> [miou, iou1,..., iouN] = " % epoch, meanIou)
                print("训练集运行结束 epoch %d Train -> f1:" % epoch, f1)
                print("训练集运行结束 epoch %d Train -> recall = " % epoch, recall)
                print("训练集运行结束 epoch %d Train -> precision = " % epoch, precision)
            if weights_path:
                SaveIndex = SaveWeight(SaveIndex, iou, weights_path, net=net)

def main():
    parser = argparse.ArgumentParser(description="default parameters")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    parser.add_argument("--category_nums", type=int, default=2, help="--")
    parser.add_argument("--total_num_Samples", type=list, default=[1, 1], help="--")
    parser.add_argument("--category_name", type=list, default=['0', '1'], help="--")
    parser.add_argument("--Saved_weight_nums", type=int, default=3, help="--")
    parser.add_argument("--learn_rate", type=int, default=0.01, help="--")
    parser.add_argument("--epoches", type=int, default=150, help="--")
    parser.add_argument("--batch_size_all", type=tuple, default=(2, 1, 1), help="--")
    parser.add_argument("--weights_path_save", type=str, default="/code/Farmland_Mapping/runs/weight/", help="--")
    args = parser.parse_args()

    base_device = args.device
    category_nums = args.category_nums
    total_num_Samples = args.total_num_Samples
    category_name = args.category_name
    Saved_weight_nums = args.Saved_weight_nums
    base_learn_rate= args.learn_rate
    base_epoches = args.epoches
    batch_size_all = args.batch_size_all
    weights_path_save = args.weights_path_save

    # 判断并创建保存权重的文件夹
    if not os.path.exists(weights_path_save):
        os.makedirs(weights_path_save)
    weights_path_save = os.path.join(weights_path_save, 'dataset_1024X1024_s128_b4')

    # 数据集
    train_val_test_path = (
        '/code/Farmland_Mapping/dataset/dataset_1024X1024_s128/',
        '/code/Farmland_Mapping/dataset/dataset_1024X1024_s128/',
        '/code/Farmland_Mapping/dataset/dataset_1024X1024_s128/',
        )
    
    dataloader_train, dataloader_val, dataloader_test = get_dataloader(
        dataset_mode=0, 
        file_path=train_val_test_path, 
        batch_size=batch_size_all
        )

    # 网络
    net = segformer.get_segformer_multiTWA(
        num_classes=2,
        phi="b1",
        pretrained=False,
        )

    net = net.to(base_device)
    
    # 优化器
    optimizer = optim.SGD(net.parameters(), lr=base_learn_rate, momentum=0.9, nesterov=True)

    # 学习率模式
    scheduler = get_learn_rate_scheduler(optimizer, 
                                         "WarmCosinDecay", 
                                         warm_up_epochs=5, 
                                         epochs_train=base_epoches)

    # 损失函数
    criterion = get_criterion(
        device=base_device, 
        lossname='segformer_3task_v1', 
        label_smoothing=0.01,
        total_num_Samples=total_num_Samples
        )
    

    # 开始训练
    train(
        device = base_device, 
        numcls = category_nums, 
        epochs_train = base_epoches, 
        net = net, 
        optimizer = optimizer, 
        criterion = criterion, 
        scheduler = scheduler, 
        SaveIndex_N = Saved_weight_nums, 
        weights_path = weights_path_save,
        category_name = category_name,
        data = (dataloader_train, dataloader_val, dataloader_test)
        )

    return net


if __name__ == "__main__":
   net = main()

