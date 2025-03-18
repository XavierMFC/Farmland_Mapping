from collections import OrderedDict
import yaml
import torch
import numpy as np
import os
from tqdm import tqdm

from evaluation.ConfusionMatrix import SegmentationMetric
from data import build_data
from models import build_model
from viz import visdomVIZ

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
    dataloader_train, dataloader_val, dataloader_test = build_data.BuildData(dataset_mode, file_path=file_path, batch_size=batch_size).get_dataset()
    return dataloader_train, dataloader_val, dataloader_test

def get_model(name, device, n_classes, weight_load=(0, None)):
    modelclass = build_model.BuildModel(name, device, n_classes, weight_load=(0, None))
    net = modelclass.get_model().to(device)
    return net


def CM2Index(CM, meanWeights, evaluation):
    iou = evaluation.meanIOU(CM)  # 单类iou
    meanIou = [(np.array(meanWeights) * np.array(iou[1:])).sum()] + iou[1:]  # 加权平均iou
    f1 = evaluation.f1(CM)
    recall = evaluation.recall(CM)
    precision = evaluation.precision(CM)
    return iou, meanIou, f1, recall, precision


def evalseg(device, dataloader_eval, net, numcls):
    """ 
    每隔固定回合测试一次
    除了写入图像、还有数据可视化
    """
    net.eval()
    evaluation = SegmentationMetric(numcls)  # 实例化, numcls表示有numcls个分类
    with torch.no_grad():
        ConfusionMatrix_test = np.zeros((numcls, numcls)).astype(int)
        for index, data in tqdm(enumerate(dataloader_eval)):
            x, y = data
            x = x.to(device)
            label = y.long().to(device)

            labels_mask = label[:, 0, ...]
            labels_edge = label[:, 1, ...]
            labels_dist = label[:, 2, ...]

            outputs_seg = net(x)

            ConfusionMatrix_test = ConfusionMatrix_test + evaluation.calConfusionMatrix(outputs_seg, labels_mask)  # 计算每个batch的混淆矩阵
            # Visual_ins.makeVizGrid(x, pre, index, epoch, p=100)
  
        iou, meanIou, f1, recall, precision = CM2Index(ConfusionMatrix_test, [.5, .5], evaluation)
        print("iou, f1, recall, precision : ", iou, meanIou, f1, recall, precision)
        print(ConfusionMatrix_test)


def main():
    device = torch.device('cuda:2' if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    category_nums = 2
    category_name = ['背景-0', '农田-1']
    weights_path_save = "/home/Garfield/savedWeight/farmlan_parcel/dpv3p/_0.859085.pt"
    
    # 数据集
    batch_size_all = (4, 1, 8)
    train_val_test_path = (
        '/home/Garfield/data/YuanHuaZhen_normaldata_15cm/tain_fd_1024_0/',
        '/home/Garfield/data/YuanHuaZhen_normaldata_15cm/val_fd_1024_0/',
        '/home/Garfield/data/YuanHuaZhen_normaldata_15cm/test_fd_1024_0/',
        )
    
    dataloader_train, dataloader_val, dataloader_test = get_dataloader(
        dataset_mode=1, 
        file_path=train_val_test_path, 
        batch_size=batch_size_all
        )
    dataset_all = {"train": dataloader_train, "val": dataloader_val, "test": dataloader_test}
    # 网络
    # net = get_model('hrnet', device, n_classes=2, weight_load=(0, None))
    
    net = get_model('deeplab_v3p', device, n_classes=2, weight_load=(0, None))


    multi_GPU = 1

    if multi_GPU:
        state_dict = torch.load(weights_path_save, map_location=device)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7::]  # remove `module.`
            new_state_dict[name] = v
        net.load_state_dict(new_state_dict)
    else:
        state_dict = torch.load(weights_path_save, map_location=device)
        net.load_state_dict(state_dict)

    for name in dataset_all.keys():
        print("%s 测试中" % name)
        data = dataset_all[name]
        evalseg(device, data, net, numcls=category_nums)

if __name__ == "__main__":
   main()

