# -*- coding: utf-8 -*-
"""
This file contains the PyTorch dataset for hyperspectral images and
related helpers.
"""
import os

import h5py
import numpy as np
import torch
import torch.utils
import torch.utils.data
from tqdm import tqdm

try:
    # Python 3
    from urllib.request import urlretrieve
except ImportError:
    # Python 2
    from urllib import urlretrieve

from utils_HSI import open_file

DATASETS_CONFIG = {
    'Houston13': {
        'img': 'Houston13.mat',
        'gt': 'Houston13_7gt.mat',
    },
    'Houston18': {
        'img': 'Houston18.mat',
        'gt': 'Houston18_7gt.mat',
    },
    'Houston13_uint8': {
        'ori_data': 'Houston13_uint8.mat',
        'gt': 'Houston13_7gt.mat',
    },
    'Houston18_uint8': {
        'ori_data': 'Houston18_uint8.mat',
        'gt': 'Houston18_7gt.mat',
    },
    'Houston13_uint8_intrR': {
        'ori_data': 'Houston13_uint8_intrR.mat',
        'gt': 'Houston13_7gt.mat',
    },
    'Houston18_uint8_intrR': {
        'ori_data': 'Houston18_uint8_intrR.mat',
        'gt': 'Houston18_7gt.mat',
    },
    'Houston13_con_smooth': {
        'ori_data': 'Houston13_con_smooth.mat',
        'gt': 'Houston13_gt_con_smooth.mat',
    },
    'paviaU': {
        'img': 'paviaU.mat',
        'gt': 'paviaU_7gt.mat',
    },
    'paviaU_con_smooth': {
        'img': 'PU_con_smooth.mat',
        'gt': 'PU_gt_con_smooth.mat',
    },
    'paviaC': {
        'img': 'paviaC.mat',
        'gt': 'paviaC_7gt.mat',
    },
    'paviaU_uint8': {
        'ori_data': 'PU_unit8.mat',
        'gt': 'paviaU_7gt.mat',
    },
    'paviaC_uint8': {
        'ori_data': 'PC_uint8.mat',
        'gt': 'paviaC_7gt.mat',
    },

    'paviaU_uint8_intrR': {
        'ori_data': 'PU_unit8_intrR.mat',
        'gt': 'paviaU_7gt.mat',
    },
    'paviaC_uint8_intrR': {
        'ori_data': 'PC_uint8_intrR.mat',
        'gt': 'paviaC_7gt.mat',
    },

    'Dioni': {
        'img': 'Dioni.mat',
        'gt': 'Dioni_gt_out68.mat',
    },
    'Loukia': {
        'img': 'Loukia.mat',
        'gt': 'Loukia_gt_out68.mat',
    },
    'SC-1': {
        'img': 'SC-1.mat',
        'gt': '1GT.mat',
    },
    'SC-3': {
        'img': 'SC-3.mat',
        'gt': '3GT.mat',
    },

    'GID_nc': {
        'img': 'GID_nc.mat',
        'gt': 'GID_nc_gt.mat',
    },

    'GID_wh': {
        'img': 'GID_wh.mat',
        'gt': 'GID_wh_gt.mat',
    },
    'HangZhou': {
        'img': 'HangZhou.mat',
        'gt': 'HangZhou_gt.mat',
    },
    'ShangHai': {
        'img': 'ShangHai.mat',
        'gt': 'ShangHai_gt.mat',
    },
}

try:
    from custom_datasets import CUSTOM_DATASETS_CONFIG

    DATASETS_CONFIG.update(CUSTOM_DATASETS_CONFIG)
except ImportError:
    pass


class TqdmUpTo(tqdm):
    """Provides `update_to(n)` which uses `tqdm.update(delta_n)`."""

    def update_to(self, b=1, bsize=1, tsize=None):
        """
        b  : int, optional
            Number of blocks transferred so far [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)  # will also set self.n = b * bsize


def get_dataset(dataset_name, target_folder="./", datasets=DATASETS_CONFIG):
    """ Gets the dataset specified by name and return the related components.
    Args:
        dataset_name: string with the name of the dataset
        target_folder (optional): folder to store the datasets, defaults to ./
        datasets (optional): dataset configuration dictionary, defaults to prebuilt one
    Returns:
        img: 3D hyperspectral image (WxHxB)
        gt: 2D int array of labels
        label_values: list of class names
        ignored_labels: list of int classes to ignore
        rgb_bands: int tuple that correspond to red, green and blue bands
    """
    palette = None

    if dataset_name not in datasets.keys():
        raise ValueError("{} dataset is unknown.".format(dataset_name))

    dataset = datasets[dataset_name]

    folder = target_folder  # + datasets[dataset_name].get('folder', dataset_name + '/')
    if dataset.get('download', False):
        # Download the dataset if is not present
        if not os.path.isdir(folder):
            os.mkdir(folder)
        for url in datasets[dataset_name]['urls']:
            # download the files
            filename = url.split('/')[-1]
            if not os.path.exists(folder + filename):
                with TqdmUpTo(unit='B', unit_scale=True, miniters=1,
                              desc="Downloading {}".format(filename)) as t:
                    urlretrieve(url, filename=folder + filename,
                                reporthook=t.update_to)
    elif not os.path.isdir(folder):
        print("WARNING: {} is not downloadable.".format(dataset_name))

    if dataset_name == 'Houston13':
        # Load the image
        # img = open_file(folder + 'Houston13.mat')['ori_data']
        Houston13_data = h5py.File('./datasets/Houston/Houston13.mat', 'r')
        img = np.transpose(Houston13_data['ori_data'])

        rgb_bands = [13, 20, 33]

        # gt = open_file(folder + 'Houston13_7gt_FillUnlabeled.mat')['map']  # 加载未标记样本局部填充结果
        # gt = open_file(folder + 'Houston13_7gt_PseudoLabel.mat')['map']  # 加载伪标签

        Houston13_7gt_data = h5py.File('./datasets/Houston/Houston13_7gt.mat', 'r')  # 加载原版标签
        gt = np.transpose(Houston13_7gt_data['map'])  # 加载原版标签

        gt = np.int64(gt)

        label_values = ["grass healthy", "grass stressed", "trees",
                        "water", "residential buildings",
                        "non-residential buildings", "road"]

        ignored_labels = [0]

    elif dataset_name == 'Houston13_uint8':
        # Load the image
        img = open_file(folder + 'Houston13_uint8.mat')['ori_data']
        # Houston13_data = h5py.File('./datasets/Houston_rgb/Houston13_rgb.mat', 'r')
        # img = np.transpose(Houston13_data['rgb'])

        rgb_bands = [13, 20, 33]

        Houston13_7gt_data = h5py.File('./datasets/Houston/Houston13_7gt.mat', 'r')  # 加载原版标签
        gt = np.transpose(Houston13_7gt_data['map'])  # 加载原版标签

        gt = np.int64(gt)

        label_values = ["grass healthy", "grass stressed", "trees",
                        "water", "residential buildings",
                        "non-residential buildings", "road"]

        ignored_labels = [0]

    elif dataset_name == 'Houston13_uint8_intrR':
        # Load the image
        img = open_file(folder + 'Houston13_intrR_uint8.mat')['ori_data']
        # Houston13_data = h5py.File('./datasets/Houston_rgb/Houston13_rgb.mat', 'r')
        # img = np.transpose(Houston13_data['rgb'])

        rgb_bands = [13, 20, 33]

        Houston13_7gt_data = h5py.File('./datasets/Houston/Houston13_7gt.mat', 'r')  # 加载原版标签
        gt = np.transpose(Houston13_7gt_data['map'])  # 加载原版标签

        gt = np.int64(gt)

        label_values = ["grass healthy", "grass stressed", "trees",
                        "water", "residential buildings",
                        "non-residential buildings", "road"]

        ignored_labels = [0]

    elif dataset_name == 'Houston13_con_smooth':
        # Load the image
        img = open_file(folder + 'Houston13_con_smooth.mat')['ori_data']
        # Houston13_data = h5py.File('./datasets/Houston_rgb/Houston13_rgb.mat', 'r')
        # img = np.transpose(Houston13_data['rgb'])

        rgb_bands = [13, 20, 33]

        gt = open_file(folder + 'Houston13_gt_con_smooth.mat')['map']  # 加载原版标签

        label_values = ["grass healthy", "grass stressed", "trees",
                        "water", "residential buildings",
                        "non-residential buildings", "road"]

        ignored_labels = [0]

    elif dataset_name == 'Houston18':
        # Load the image
        # img = open_file(folder + 'Houston18.mat')['ori_data']
        Houston18_data = h5py.File('./datasets/Houston/Houston18.mat', 'r')
        img = np.transpose(Houston18_data['ori_data'])

        rgb_bands = [13, 20, 33]

        # gt = open_file(folder + 'Houston18_7gt.mat')['map']
        Houston18_7gt_data = h5py.File('./datasets/Houston/Houston18_7gt.mat', 'r')
        gt = np.transpose(Houston18_7gt_data['map'])
        gt = np.int64(gt)

        label_values = ["grass healthy", "grass stressed", "trees",
                        "water", "residential buildings",
                        "non-residential buildings", "road"]

        ignored_labels = [0]

    elif dataset_name == 'Houston18_uint8':
        # Load the image
        img = open_file(folder + 'Houston18_uint8.mat')['ori_data']
        # Houston18_data = h5py.File('./datasets/Houston/Houston18.mat', 'r')
        # img = np.transpose(Houston18_data['ori_data'])

        rgb_bands = [13, 20, 33]

        # gt = open_file(folder + 'Houston18_7gt.mat')['map']
        Houston18_7gt_data = h5py.File('./datasets/Houston/Houston18_7gt.mat', 'r')
        gt = np.transpose(Houston18_7gt_data['map'])
        gt = np.int64(gt)

        label_values = ["grass healthy", "grass stressed", "trees",
                        "water", "residential buildings",
                        "non-residential buildings", "road"]

        ignored_labels = [0]

    elif dataset_name == 'Houston18_uint8_intrR':
        # Load the image
        img = open_file(folder + 'Houston18_intrR_uint8.mat')['ori_data']
        # Houston18_data = h5py.File('./datasets/Houston/Houston18.mat', 'r')
        # img = np.transpose(Houston18_data['ori_data'])

        rgb_bands = [13, 20, 33]

        # gt = open_file(folder + 'Houston18_7gt.mat')['map']
        Houston18_7gt_data = h5py.File('./datasets/Houston/Houston18_7gt.mat', 'r')
        gt = np.transpose(Houston18_7gt_data['map'])
        gt = np.int64(gt)

        label_values = ["grass healthy", "grass stressed", "trees",
                        "water", "residential buildings",
                        "non-residential buildings", "road"]

        ignored_labels = [0]

    elif dataset_name == 'paviaU':
        # Load the image
        img = open_file(folder + 'paviaU.mat')['ori_data']

        rgb_bands = [20, 30, 30]

        # gt = open_file(folder + 'paviaU_7gt_FillUnlabeled.mat')['map']  # 加载未标记样本局部填充结果
        # gt = open_file(folder + 'paviaU_7gt_PseudoLabel.mat')['map'] # 加载伪标签
        gt = open_file(folder + 'paviaU_7gt.mat')['map']  # 加载原版标签

        label_values = ["tree", "asphalt", "brick",
                        "bitumen", "shadow", 'meadow', 'bare soil']

        ignored_labels = [0]

    elif dataset_name == 'paviaU_con_smooth':
        # Load the image
        img = open_file(folder + 'PU_con_smooth.mat')['ori_data']

        rgb_bands = [20, 30, 30]

        # gt = open_file(folder + 'paviaU_7gt_FillUnlabeled.mat')['map']  # 加载未标记样本局部填充结果
        # gt = open_file(folder + 'paviaU_7gt_PseudoLabel.mat')['map'] # 加载伪标签
        gt = open_file(folder + 'PU_gt_con_smooth.mat')['map']  # 加载原版标签

        label_values = ["tree", "asphalt", "brick",
                        "bitumen", "shadow", 'meadow', 'bare soil']

        ignored_labels = [0]

    elif dataset_name == 'paviaC':
        # Load the image
        img = open_file(folder + 'paviaC.mat')['ori_data']

        rgb_bands = [20, 30, 30]

        gt = open_file(folder + 'paviaC_7gt.mat')['map']

        label_values = ["tree", "asphalt", "brick",
                        "bitumen", "shadow", 'meadow', 'bare soil']

        ignored_labels = [0]

    elif dataset_name == 'paviaU_uint8':
        # Load the image
        img = open_file(folder + 'PU_uint8.mat')['ori_data']

        rgb_bands = [20, 30, 30]

        # gt = open_file(folder + 'paviaU_7gt_FillUnlabeled.mat')['map']  # 加载未标记样本局部填充结果
        # gt = open_file(folder + 'paviaU_7gt_PseudoLabel.mat')['map'] # 加载伪标签
        gt = open_file(folder + 'paviaU_7gt.mat')['map']  # 加载原版标签

        label_values = ["tree", "asphalt", "brick",
                        "bitumen", "shadow", 'meadow', 'bare soil']

        ignored_labels = [0]

    elif dataset_name == 'paviaC_uint8':
        # Load the image
        img = open_file(folder + 'PC_uint8.mat')['ori_data']

        rgb_bands = [20, 30, 30]

        gt = open_file(folder + 'paviaC_7gt.mat')['map']

        label_values = ["tree", "asphalt", "brick",
                        "bitumen", "shadow", 'meadow', 'bare soil']

        ignored_labels = [0]

    elif dataset_name == 'paviaU_uint8_intrR':
        # Load the image
        img = open_file(folder + 'PU_uint8_intrR.mat')['ori_data']

        rgb_bands = [20, 30, 30]

        # gt = open_file(folder + 'paviaU_7gt_FillUnlabeled.mat')['map']  # 加载未标记样本局部填充结果
        # gt = open_file(folder + 'paviaU_7gt_PseudoLabel.mat')['map'] # 加载伪标签
        gt = open_file(folder + 'paviaU_7gt.mat')['map']  # 加载原版标签

        label_values = ["tree", "asphalt", "brick",
                        "bitumen", "shadow", 'meadow', 'bare soil']

        ignored_labels = [0]

    elif dataset_name == 'paviaC_uint8_intrR':
        # Load the image
        img = open_file(folder + 'PC_uint8_intrR.mat')['ori_data']

        rgb_bands = [20, 30, 30]

        gt = open_file(folder + 'paviaC_7gt.mat')['map']

        label_values = ["tree", "asphalt", "brick",
                        "bitumen", "shadow", 'meadow', 'bare soil']

        ignored_labels = [0]

    elif dataset_name == 'Loukia':
        # Load the image
        img = open_file(folder + 'Loukia.mat')['ori_data']

        rgb_bands = [20, 30, 30]

        gt = open_file(folder + 'Loukia_gt_out68.mat')['map']

        label_values = ["Dense Urban Fabric", "Mineral Extraction Sites", "Non Irrigated Arable Land",
                        "Fruit Trees", "Olive Groves", 'Coniferous Forest', 'Dense Sderophyllous Vegetation',
                        'Sparce Sderophyllous Vegetation', 'Sparcely Vegetated Areas', 'Rocks and Sand', 'Water',
                        'Coastal Water']

        ignored_labels = [0]

    elif dataset_name == 'Dioni':
        # Load the image
        img = open_file(folder + 'Dioni.mat')['ori_data']

        rgb_bands = [20, 30, 30]

        gt = open_file(folder + 'Dioni_gt_out68.mat')['map']

        label_values = ["Dense Urban Fabric", "Mineral Extraction Sites", "Non Irrigated Arable Land",
                        "Fruit Trees", "Olive Groves", 'Coniferous Forest', 'Dense Sderophyllous Vegetation',
                        'Sparce Sderophyllous Vegetation', 'Sparcely Vegetated Areas', 'Rocks and Sand', 'Water',
                        'Coastal Water']

        ignored_labels = [0]

    elif dataset_name == 'SC-1':
        # Load the image
        img = open_file(folder + 'SC-1.mat')['image']

        rgb_bands = [20, 30, 30]

        gt = open_file(folder + '1GT.mat')['GT']

        label_values = ['1', '2', '3', '4', '5', '6']

        ignored_labels = [0]

    elif dataset_name == 'SC-3':
        # Load the image
        img = open_file(folder + 'SC-3.mat')['image']

        rgb_bands = [20, 30, 30]

        # gt = open_file(folder + '3GT.mat')['GT']
        _3GT = h5py.File('./datasets/Qingdao/3GT.mat', 'r')
        gt = np.transpose(_3GT['GT'])
        gt = np.int64(gt)

        label_values = ['1', '2', '3', '4', '5', '6']

        ignored_labels = [0]

    elif dataset_name == 'GID_nc':
        # Load the image
        _img = h5py.File('./datasets/GID/GID_nc.mat', 'r')  # 该数据集第四个通道全是255
        img = np.transpose(_img['img'])

        rgb_bands = [20, 30, 30]

        _3GT = h5py.File('./datasets/GID/GID_nc_gt.mat', 'r')
        gt = np.transpose(_3GT['GT'])
        gt = np.int64(gt)

        label_values = ['1', '2', '3', '4', '5']

        ignored_labels = [0]

    elif dataset_name == 'GID_wh':  # 该数据集第一个通道全是255，GID_wh 和 GID_nc 通道不匹配
        # Load the image
        _img = h5py.File('./datasets/GID/GID_wh.mat', 'r')
        img = np.transpose(_img['img'])

        rgb_bands = [20, 30, 30]

        _3GT = h5py.File('./datasets/GID/GID_wh_gt.mat', 'r')
        gt = np.transpose(_3GT['GT'])
        gt = np.int64(gt)

        label_values = ['1', '2', '3', '4', '5']

        ignored_labels = [0]

    elif dataset_name == 'HangZhou':
        # Load the image
        img = open_file(folder + 'HangZhou.mat')['DataCube2']

        rgb_bands = [20, 30, 30]

        gt = open_file(folder + 'HangZhou_gt.mat')['gt2']

        label_values = ['1', '2', '3']

        ignored_labels = [0]

    elif dataset_name == 'ShangHai':
        # Load the image
        img = open_file(folder + 'ShangHai.mat')['DataCube1']

        rgb_bands = [20, 30, 30]

        gt = open_file(folder + 'ShangHai_gt.mat')['gt1']

        label_values = ['1', '2', '3']

        ignored_labels = [0]
    else:
        # Custom dataset
        img, gt, rgb_bands, ignored_labels, label_values, palette = CUSTOM_DATASETS_CONFIG[dataset_name]['loader'](
            folder)

    # Filter NaN out
    nan_mask = np.isnan(img.sum(axis=-1))
    if np.count_nonzero(nan_mask) > 0:
        print(
            "Warning: NaN have been found in the data. It is preferable to remove them beforehand. Learning on NaN data is disabled.")
    img[nan_mask] = 0
    gt[nan_mask] = 0
    ignored_labels.append(0)

    ignored_labels = list(set(ignored_labels))
    # Normalization https://blog.csdn.net/qq_35091353/article/details/111768094
    # 每个样本用最大值归一化后缩放到单位范数（2）
    img = np.asarray(img, dtype='float32')  # 注意这里把原本的双精度降为单精度

    m, n, d = img.shape[0], img.shape[1], img.shape[2]
    img = img.reshape((m * n, -1))
    img = img / img.max()

    img_temp = np.sqrt(np.asarray((img ** 2).sum(1)))
    img_temp = np.expand_dims(img_temp, axis=1)
    img_temp = img_temp.repeat(d, axis=1)
    img_temp[img_temp == 0] = 1

    # img_temp = 1  # 特别地，不进行L2归一化，保证无损

    img = img / img_temp
    img = np.reshape(img, (m, n, -1))

    # 反转光谱反射率
    # img = - img + 1

    return img, gt, label_values, ignored_labels, rgb_bands, palette


class HyperX(torch.utils.data.Dataset):
    """ Generic class for a hyperspectral scene """

    def __init__(self, data, gt, transform=None, **hyperparams):
        """
        Args:
            data: 3D hyperspectral image
            gt: 2D array of labels
            patch_size: int, size of the spatial neighbourhood
            center_pixel: bool, set to True to consider only the label of the
                          center pixel
            data_augmentation: bool, set to True to perform random flips
            supervision: 'full' or 'semi' supervised algorithms
        """
        super(HyperX, self).__init__()
        self.transform = transform
        self.data = data
        self.label = gt
        self.patch_size = hyperparams['patch_size']
        self.ignored_labels = set(hyperparams['ignored_labels'])
        self.flip_augmentation = hyperparams['flip_augmentation']
        self.radiation_augmentation = hyperparams['radiation_augmentation']
        self.mixture_augmentation = hyperparams['mixture_augmentation']
        self.center_pixel = hyperparams['center_pixel']
        supervision = hyperparams['supervision']
        # Fully supervised : use all pixels with label not ignored
        if supervision == 'full':
            mask = np.ones_like(gt)
            for l in self.ignored_labels:
                mask[gt == l] = 0
        # Semi-supervised : use all pixels, except padding
        elif supervision == 'semi':
            mask = np.ones_like(gt)
        x_pos, y_pos = np.nonzero(mask)
        p = self.patch_size // 2
        self.indices = np.array([(x, y) for x, y in zip(x_pos, y_pos) if
                                 x > p and x < data.shape[0] - p and y > p and y < data.shape[1] - p])
        self.labels = [self.label[x, y] for x, y in self.indices]

        state = np.random.get_state()
        np.random.shuffle(self.indices)
        np.random.set_state(state)
        np.random.shuffle(self.labels)

    @staticmethod
    def flip(*arrays):
        horizontal = np.random.random() > 0.5
        vertical = np.random.random() > 0.5
        if horizontal:
            arrays = [np.fliplr(arr) for arr in arrays]
        if vertical:
            arrays = [np.flipud(arr) for arr in arrays]
        return arrays

    @staticmethod
    def radiation_noise(data, alpha_range=(0.9, 1.1), beta=1 / 25):
        alpha = np.random.uniform(*alpha_range)
        noise = np.random.normal(loc=0., scale=1.0, size=data.shape)
        return alpha * data + beta * noise

    def mixture_noise(self, data, label, beta=1 / 25):
        alpha1, alpha2 = np.random.uniform(0.01, 1., size=2)
        noise = np.random.normal(loc=0., scale=1.0, size=data.shape)
        data2 = np.zeros_like(data)
        for idx, value in np.ndenumerate(label):
            if value not in self.ignored_labels:
                l_indices = np.nonzero(self.labels == value)[0]
                l_indice = np.random.choice(l_indices)
                assert (self.labels[l_indice] == value)
                x, y = self.indices[l_indice]
                data2[idx] = self.data[x, y]
        return (alpha1 * data + alpha2 * data2) / (alpha1 + alpha2) + beta * noise

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        x, y = self.indices[i]
        x1, y1 = x - self.patch_size // 2, y - self.patch_size // 2
        x2, y2 = x1 + self.patch_size, y1 + self.patch_size

        data = self.data[x1:x2, y1:y2]
        label = self.label[x1:x2, y1:y2]

        if self.flip_augmentation and self.patch_size > 1 and np.random.random() < 0.5:
            # Perform data augmentation (only on 2D patches)
            data, label = self.flip(data, label)
        if self.radiation_augmentation and np.random.random() < 0.5:
            data = self.radiation_noise(data)
        if self.mixture_augmentation and np.random.random() < 0.5:
            data = self.mixture_noise(data, label)

        # Copy the data into numpy arrays (PyTorch doesn't like numpy views)
        data = np.asarray(np.copy(data).transpose((2, 0, 1)), dtype='float32')
        label = np.asarray(np.copy(label), dtype='int64')

        # Load the data into PyTorch tensors
        data = torch.from_numpy(data)
        label = torch.from_numpy(label)
        # Extract the center label if needed
        if self.center_pixel and self.patch_size > 1:
            label = label[self.patch_size // 2, self.patch_size // 2]
        # Remove unused dimensions when we work with invidual spectrums
        elif self.patch_size == 1:
            data = data[:, 0, 0]
            label = label[0, 0]
        else:
            label = self.labels[i]

        # Add a fourth dimension for 3D CNN
        # if self.patch_size > 1:
        #     # Make 4D data ((Batch x) Planes x Channels x Width x Height)
        #     data = data.unsqueeze(0)
        # plt.imshow(data[[10,23,23],:,:].permute(1,2,0))
        # plt.show()
        return data, label


class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.data, self.label = next(self.loader)

        except StopIteration:
            self.next_input = None

            return
        with torch.cuda.stream(self.stream):
            self.data = self.data.cuda(non_blocking=True)
            self.label = self.label.cuda(non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        data = self.data
        label = self.label

        self.preload()
        return data, label
