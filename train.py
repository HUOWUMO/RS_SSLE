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

from con_losses import SupConLoss
from datasets import get_dataset, HyperX
from network import discriminator
from network import generator
from utils_HSI import sample_gt, metrics, seed_worker

parser = argparse.ArgumentParser(description='PyTorch SDEnet')
parser.add_argument('--save_path', type=str, default='./results/')
parser.add_argument('--data_path', type=str, default='./datasets/Houston/')

parser.add_argument('--source_name', type=str, default='Houston13',  # Dioni
                    help='the name of the source dir')
parser.add_argument('--target_name', type=str, default='Houston18',  # Loukia
                    help='the name of the test dir')
parser.add_argument('--gpu', type=int, default=0,
                    help="Specify CUDA device (defaults to -1, which learns on CPU)")

group_train = parser.add_argument_group('Training')  # 超参数写在这里可以在epoch中打印
group_train.add_argument('--patch_size', type=int, default=13,
                         help="Size of the spatial neighbourhood (optional, if ""absent will be set by the model)")

group_train.add_argument('--lr', type=float, default=0.001,
                         help="Learning rate, set by the model if not specified.")
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum (default: 0.5)')
group_train.add_argument('--batch_size', type=int, default=64,
                         help="Batch size (optional, if absent will be set by the model")
group_train.add_argument('--pro_dim', type=int, default=128)
group_train.add_argument('--std', type=int, default=7)
group_train.add_argument('--init_alpha', type=float, default=0.9)

group_train.add_argument('--ni', type=str, default='0.1')
group_train.add_argument('--beta', type=str, default='0.5')
group_train.add_argument('--N', type=str, default='3')

group_train.add_argument('--dim', type=int, default=64)
group_train.add_argument('--test_stride', type=int, default=1,
                         help="Sliding window step stride during inference (default = 1)")
parser.add_argument('--seed', type=int, default=233,
                    help='random seed ')
parser.add_argument('--l2_decay', type=float, default=1e-4,
                    help='the L2  weight decay')
parser.add_argument('--training_sample_ratio', type=float, default=0.8,
                    help='training sample ratio')
parser.add_argument('--re_ratio', type=int, default=5,  # PaviaU-1 Houston13-5，图像扩充倍数
                    help='multiple of of data augmentation')
parser.add_argument('--max_epoch', type=int, default=200)
parser.add_argument('--log_interval', type=int, default=40)

parser.add_argument('--lambda_1', type=float, default=1.0)
parser.add_argument('--lambda_2', type=float, default=0.5)
parser.add_argument('--alpha_1', type=float, default=1.0)
parser.add_argument('--alpha_2', type=float, default=1.0)
parser.add_argument('--lr_scheduler', type=str, default='none')

# 注意，再SDEnet里，只有HOUSTON数据经过了flip与illumination增强,增强后，训练时间几乎翻了三倍
group_da = parser.add_argument_group('Data augmentation')
group_da.add_argument('--flip_augmentation', action='store_true', default=False,
                      # 命令中加上--flip_augmentation触发True, 不加表示False
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
    t3 = time.time()
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
    t4 = time.time()
    if tgt:
        results = metrics(ps, ys, n_classes=ys.max() + 1)  # metrics 和 show_results 均是可直接使用的HSI计算工具
        print(results['Confusion_matrix'], '\n', 'TPR:', np.round(results['TPR'] * 100, 2), '\n', 'OA:',
              np.round(results['Accuracy'], 2), 'AA:', np.round(100 * results['AA'], 2), 'Kappa:',
              np.round(100 * results["Kappa"], 2))
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

    # 数据集类别比例(1-t%)
    tabulate_PU = [1 - 0.0779, 1 - 0.1686, 1 - 0.0936, 1 - 0.0338, 1 - 0.0241, 1 - 0.4741, 1 - 0.1279]
    tabulate_H13 = [1 - 0.1364, 1 - 0.1443, 1 - 0.1443, 1 - 0.1126, 1 - 0.1261, 1 - 0.1613, 1 - 0.1751]

    if args.source_name == 'Houston13':
        img_src_synthesis = np.load("datasets/Houston/img_src_synthesis_H13.npy")
        alpha = tabulate_H13
    elif args.source_name == 'paviaU':
        img_src_synthesis = np.load("datasets/Pavia/img_src_synthesis_PU.npy")
        alpha = tabulate_PU
    else:
        img_src_synthesis = np.load("datasets/Houston/img_src_synthesis_H13.npy")
        alpha = tabulate_H13

    hyperparams.update({'n_classes': num_classes, 'n_bands': N_BANDS, 'ignored_labels': IGNORED_LABELS,
                        'device': args.gpu, 'center_pixel': None, 'supervision': 'full'})

    r = int(hyperparams['patch_size'] / 2) + 1
    img_src = np.pad(img_src, ((r, r), (r, r), (0, 0)), 'symmetric')
    img_src_synthesis = np.pad(img_src_synthesis, ((r, r), (r, r), (0, 0)), 'symmetric')
    img_tar = np.pad(img_tar, ((r, r), (r, r), (0, 0)), 'symmetric')
    gt_src = np.pad(gt_src, ((r, r), (r, r)), 'constant', constant_values=(0, 0))
    gt_tar = np.pad(gt_tar, ((r, r), (r, r)), 'constant', constant_values=(0, 0))

    train_gt_src, val_gt_src, _, _ = sample_gt(gt_src, args.training_sample_ratio, mode='random')  # 划分训练集和验证集保持了类别比例
    test_gt_tar, _, _, _ = sample_gt(gt_tar, 1, mode='random')
    img_src_con, train_gt_src_con = img_src, train_gt_src
    img_src_con_synthesis, train_gt_src_con = img_src_synthesis, train_gt_src
    val_gt_src_con = val_gt_src
    if tmp < 1:  # 如果预计增广后的训练样本数量少于测试样本数量，才真的对训练样本+验证样本增广
        for i in range(args.re_ratio - 1):
            img_src_con = np.concatenate((img_src_con, img_src))
            img_src_con_synthesis = np.concatenate((img_src_con_synthesis, img_src))
            train_gt_src_con = np.concatenate((train_gt_src_con, train_gt_src))
            val_gt_src_con = np.concatenate((val_gt_src_con, val_gt_src))

    hyperparams_train = hyperparams.copy()
    g = torch.Generator()
    g.manual_seed(args.seed)

    # 这里沿光谱方向拼接到原始data后面,由于radiation_augmentation的存在，拼接后size的改变会造成训练数据的改变
    train_dataset = HyperX(np.concatenate((img_src_con, img_src_con_synthesis), axis=2),
                           train_gt_src_con,
                           **hyperparams_train)

    train_loader = data.DataLoader(train_dataset,
                                   batch_size=hyperparams['batch_size'],
                                   pin_memory=True,
                                   worker_init_fn=seed_worker,
                                   generator=g,
                                   shuffle=True, )
    # 注意，验证的数据是原始数据
    val_dataset = HyperX(img_src_con, val_gt_src_con, **hyperparams)
    val_loader = data.DataLoader(val_dataset,
                                 pin_memory=True,
                                 batch_size=hyperparams['batch_size'])
    test_dataset = HyperX(img_tar, test_gt_tar, **hyperparams)
    test_loader = data.DataLoader(test_dataset,
                                  pin_memory=True,
                                  worker_init_fn=seed_worker,
                                  generator=g,
                                  batch_size=hyperparams['batch_size'])
    imsize = [hyperparams['patch_size'], hyperparams['patch_size']]

    # =================鉴别器模型配置=================

    D_net = discriminator.Discriminator_AddDim(inchannel=N_BANDS, outchannel=args.pro_dim, dim=args.dim,
                                               num_classes=num_classes,
                                               patch_size=hyperparams['patch_size']).to(args.gpu)

    G_net = generator.Generator_Gaussian(inchannel=N_BANDS, patch_size=hyperparams['patch_size'], std=args.std, re_grad_weight=False, init_alpha=args.init_alpha, re_grad_alpha=False).to(args.gpu)

    # =================模型LOSS与训练优化配置=================
    D_opt = optim.Adam(D_net.parameters(), lr=args.lr)
    cls_criterion = nn.CrossEntropyLoss()
    con_criterion = SupConLoss(device=args.gpu)

    # =================开始训练=================
    best_acc = 0
    taracc, taracc_list = 0, []
    for epoch in range(1, args.max_epoch + 1):

        t1 = time.time()
        loss_list = []
        D_net.train()
        for i, (x_double, y) in enumerate(train_loader):
            x_double, y = x_double.to(args.gpu), y.to(args.gpu)
            y = y - 1
            x = x_double[:, 0:N_BANDS, :, :]
            x_synthesis = x_double[:, N_BANDS:N_BANDS * 2, :, :]
          
            p_SD, z_SD = D_net(x, mode='train')
            p_GD, z_GD = D_net(x_GD, mode='train')
            p_synthesis, z_synthesis = D_net(x_synthesis, mode='train')

            # 交叉熵保证精度，这里严格遵循VRM原则，采用三组交叉熵
            src_cls_loss = cls_criterion(p_SD, y.long()) + cls_criterion(x_synthesis, y.long()) + cls_criterion(p_GD,
                                                                                                             y.long())

            # 对比学习实现特征聚类
            # 拼在一起的三组特征向量，共享一组标签
            zsrc = torch.cat([z_SD.unsqueeze(1), z_GD.unsqueeze(1), z_synthesis.unsqueeze(1)], dim=1)
            con_loss = con_criterion(zsrc, y, adv=False)

            loss1 = src_cls_loss + args.lambda_1 * con_loss
            D_opt.zero_grad()
            loss1.backward()
            D_opt.step()

            loss_list.append([src_cls_loss.item(), con_loss.item()])

        src_cls_loss, con_loss = np.mean(loss_list, 0)

        D_net.eval()
        teacc = evaluate(D_net, val_loader, args.gpu)
        if best_acc <= teacc:
            best_acc = teacc
            torch.save({'Discriminator': D_net.state_dict()}, os.path.join(log_dir, f'best.pkl'))
            torch.save({'Generator': G_net.state_dict()}, os.path.join(log_dir, f'best_G.pkl'))
        t2 = time.time()

        print(
            f'epoch {epoch}, train {len(train_loader.dataset)}, time {t2 - t1:.2f}, src_cls {src_cls_loss:.4f} con {con_loss:.4f} /// val {len(val_loader.dataset)}, teacc {teacc:2.2f}')
        writer.add_scalar('src_cls_loss', src_cls_loss, epoch)
        writer.add_scalar('con_loss', con_loss, epoch)
        writer.add_scalar('teacc', teacc, epoch)

        if epoch % args.log_interval == 0:
            pklpath = f'{log_dir}/best.pkl'
            taracc = evaluate_tgt(D_net, args.gpu, test_loader, pklpath)
            taracc_list.append(round(taracc, 2))
            print(f'load pth, target sample number {len(test_loader.dataset)}, max taracc {max(taracc_list):2.2f}')
    writer.close()


if __name__ == '__main__':
    experiment()
