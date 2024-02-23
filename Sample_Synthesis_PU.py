import argparse
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as data
from tensorboardX import SummaryWriter
from torch import optim
import scipy.io as sio
from con_losses import SupConLoss
from datasets import get_dataset, HyperX
from network import discriminator
from network import generator
from utils_HSI import sample_gt, metrics, seed_worker
from unbalanced_loss.focal_loss import MultiFocalLoss
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='PyTorch SDEnet')
parser.add_argument('--save_path', type=str, default='./results/')
parser.add_argument('--data_path', type=str, default='./datasets/Pavia/')

parser.add_argument('--source_name', type=str, default='paviaU',
                    help='the name of the source dir')
parser.add_argument('--target_name', type=str, default='paviaC',
                    help='the name of the test dir')
parser.add_argument('--gpu', type=int, default=0,
                    help="Specify CUDA device (defaults to -1, which learns on CPU)")

group_train = parser.add_argument_group('Training')
group_train.add_argument('--patch_size', type=int, default=13,
                         help="Size of the spatial neighbourhood (optional, if ""absent will be set by the model)")

group_train.add_argument('--lr', type=float, default=0.001,
                         help="Learning rate, set by the model if not specified.")
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum (default: 0.5)')
group_train.add_argument('--batch_size', type=int, default=64,
                         help="Batch size (optional, if absent will be set by the model")
group_train.add_argument('--pro_dim', type=int, default=128)
group_train.add_argument('--test_stride', type=int, default=1,
                         help="Sliding window step stride during inference (default = 1)")
parser.add_argument('--seed', type=int, default=233,
                    help='random seed ')
parser.add_argument('--l2_decay', type=float, default=1e-4,
                    help='the L2  weight decay')
parser.add_argument('--training_sample_ratio', type=float, default=0.8,
                    help='training sample ratio')
parser.add_argument('--re_ratio', type=int, default=1,  # PaviaU-1 Houston13-5，图像扩充倍数
                    help='multiple of of data augmentation')
parser.add_argument('--max_epoch', type=int, default=400)
parser.add_argument('--log_interval', type=int, default=40)

parser.add_argument('--lambda_1', type=float, default=1.0)
parser.add_argument('--lambda_2', type=float, default=1.0)
parser.add_argument('--lr_scheduler', type=str, default='none')

# 注意，再SDEnet里，只有HOUSTON数据经过了flip与illumination增强
group_da = parser.add_argument_group('Data augmentation')
group_da.add_argument('--flip_augmentation', action='store_true', default=False,
                      help="Random flips (if patch_size > 1)")
group_da.add_argument('--radiation_augmentation', action='store_true', default=False,
                      help="Random radiation noise (illumination)")
group_da.add_argument('--mixture_augmentation', action='store_true', default=False,
                      help="Random mixes between spectra")
args = parser.parse_args()


def \
        evaluate(net, val_loader, gpu, tgt=False):
    ps = []
    ys = []
    for i, (x1, y1) in enumerate(val_loader):
        y1 = y1 - 1
        with torch.no_grad():
            x1 = x1.to(gpu)
            p1 = net(x1)
            p1 = p1.argmax(dim=1)
            ps.append(p1.detach().cpu().numpy())
            ys.append(y1.numpy())
    ps = np.concatenate(ps)
    ys = np.concatenate(ys)
    acc = np.mean(ys == ps) * 100
    if tgt:
        results = metrics(ps, ys, n_classes=ys.max() + 1)  # metrics 和 show_results 均是可直接使用的HSI计算工具
        print(results['Confusion_matrix'], '\n', 'TPR:', np.round(results['TPR'] * 100, 2), '\n', 'OA:',
              results['Accuracy'], 'Kappa:', results["Kappa"])
    return acc


def evaluate_tgt(cls_net, gpu, loader, modelpath):
    saved_weight = torch.load(modelpath)
    cls_net.load_state_dict(saved_weight['Discriminator'])
    cls_net.eval()
    teacc = evaluate(cls_net, loader, gpu, tgt=True)
    return teacc


def experiment():
    settings = locals().copy()
    print(settings)
    hyperparams = vars(args)
    print(hyperparams)
    now_time = datetime.now()
    time_str = datetime.strftime(now_time, '%m-%d_%H-%M-%S')
    root = os.path.join(args.save_path, args.source_name + 'to' + args.target_name)
    log_dir = os.path.join(root, str(args.lr) + '_dim' + str(args.pro_dim) +
                           '_pt' + str(args.patch_size) + '_bs' + str(args.batch_size) + '_' + time_str)
    if not os.path.exists(root):
        os.makedirs(root)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    writer = SummaryWriter(log_dir)
    df = pd.DataFrame([args])
    df.to_csv(os.path.join(log_dir, 'params.txt'))

    seed_worker(args.seed)
    img_src, gt_src, LABEL_VALUES_src, IGNORED_LABELS, RGB_BANDS, palette = get_dataset(args.source_name,
                                                                                        args.data_path)
    img_tar, gt_tar, LABEL_VALUES_tar, IGNORED_LABELS, RGB_BANDS, palette = get_dataset(args.target_name,
                                                                                        args.data_path)

    sample_num_src = len(np.nonzero(gt_src)[0])
    sample_num_tar = len(np.nonzero(gt_tar)[0])

    tmp = args.training_sample_ratio * args.re_ratio * sample_num_src / sample_num_tar
    num_classes = gt_src.max()
    N_BANDS = img_src.shape[-1]

    m = img_src.shape[0]
    n = img_src.shape[1]
    img_src_synthesis = np.zeros((m, n, N_BANDS))

    S_net = generator.Synthesis(N=N_BANDS, ni=0.1, beta=0.5, num_iteration=3).to(args.gpu)

    for i in range(0, m):
        for j in range(0, n):
            img_src_synthesis[i, j, :] = S_net(img_src[i, j, :])

    # 保存一次
    np.save("datasets/Pavia/img_src_synthesis_PU.npy", img_src_synthesis)

    # =============可视化合成结果=============
    # 随机取一个位置
    i = 77
    j = 220

    img_src_synthesis[i, j, :] = S_net(img_src[i, j, :])

    x_axis = range(0, N_BANDS)  # 左闭右开
    y_axis_src = img_src[i, j, :]
    y_axis_src_synthesis = img_src_synthesis[i, j, :]

    # 画图
    plt.plot(x_axis, y_axis_src, label='Original Spectral Reflectance')
    plt.plot(x_axis, y_axis_src_synthesis, label='MRF Spectral Synthesis')

    plt.legend()
    # 添加标题和标签
    plt.title('Pixel From PaviaU')
    plt.xlabel('spectral band')
    plt.ylabel('spectral reflectance')
    # 显示图像
    plt.show()  # 数据集采用正则化归一化处理


if __name__ == '__main__':
    experiment()