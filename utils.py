import os
import logging
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
from torch.autograd import Variable
import torch.nn.functional as F
import random
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
import copy
from functools import reduce

from models import *
from datasets import (
    MNIST_truncated,
    CIFAR10_truncated,
    CIFAR100_truncated,
    ImageFolder_custom,
    SVHN_custom,
    FashionMNIST_truncated,
    CustomTensorDataset,
    CelebA_custom,
    FEMNIST,
    Generated,
    genData,
    CheXpertDataset,
    TransformTwice,
    OfficeDataset,
)
from math import sqrt

import torch.nn as nn
from torch import Tensor
import torch.optim as optim
import torchvision.utils as vutils
import time
import random
from typing import Sequence, Dict, Any, Set, Tuple, Optional
from abc import ABC, abstractmethod

# from models.mnist_model import Generator, Discriminator, DHead, QHead
# from config import params
# import sklearn.datasets as sk
from sklearn.datasets import load_svmlight_file

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def mkdirs(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass


def load_mnist_data(datadir):

    transform = transforms.Compose([transforms.ToTensor()])

    mnist_train_ds = MNIST_truncated(
        datadir, train=True, download=True, transform=transform
    )
    mnist_test_ds = MNIST_truncated(
        datadir, train=False, download=True, transform=transform
    )

    X_train, y_train = mnist_train_ds.data, mnist_train_ds.target
    X_test, y_test = mnist_test_ds.data, mnist_test_ds.target

    X_train = X_train.data.numpy()
    y_train = y_train.data.numpy()
    X_test = X_test.data.numpy()
    y_test = y_test.data.numpy()

    return (X_train, y_train, X_test, y_test)


def load_fmnist_data(datadir):

    transform = transforms.Compose([transforms.ToTensor()])

    mnist_train_ds = FashionMNIST_truncated(
        datadir, train=True, download=True, transform=transform
    )
    mnist_test_ds = FashionMNIST_truncated(
        datadir, train=False, download=True, transform=transform
    )

    X_train, y_train = mnist_train_ds.data, mnist_train_ds.target
    X_test, y_test = mnist_test_ds.data, mnist_test_ds.target

    X_train = X_train.data.numpy()
    y_train = y_train.data.numpy()
    X_test = X_test.data.numpy()
    y_test = y_test.data.numpy()

    return (X_train, y_train, X_test, y_test)


def prepare_data(args):
    data_base_path = "./data"
    transform_office = transforms.Compose(
        [
            transforms.Resize([256, 256]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation((-30, 30)),
            transforms.ToTensor(),
        ]
    )

    strong_trans = transforms.Compose(
        [
            transforms.Resize([256, 256]),  # 更大范围的裁剪
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        0.4, 0.4, 0.4, 0.1
                    )  # 随机调整亮度、对比度、饱和度和色调
                ],
                p=0.8,
            ),  # 以较高概率应用颜色抖动
            transforms.RandomGrayscale(p=0.2),  # 20% 概率将图像转为灰度
            transforms.ToTensor(),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.Resize([256, 256]),
            transforms.ToTensor(),
        ]
    )

    unlabel_trans = TransformTwice(transform_office, strong_trans)

    # amazon
    amazon_trainset = OfficeDataset(
        data_base_path, "amazon", transform=transform_office
    )
    amazon_testset = OfficeDataset(
        data_base_path, "amazon", transform=transform_test, train=False
    )
    # caltech
    caltech_trainset = OfficeDataset(data_base_path, "caltech", transform=unlabel_trans)
    caltech_testset = OfficeDataset(
        data_base_path, "caltech", transform=transform_test, train=False
    )
    # dslr
    dslr_trainset = OfficeDataset(data_base_path, "dslr", transform=unlabel_trans)
    dslr_testset = OfficeDataset(
        data_base_path, "dslr", transform=transform_test, train=False
    )
    # webcam
    webcam_trainset = OfficeDataset(data_base_path, "webcam", transform=unlabel_trans)
    webcam_testset = OfficeDataset(
        data_base_path, "webcam", transform=transform_test, train=False
    )

    amazon_train_loader = torch.utils.data.DataLoader(
        amazon_trainset, batch_size=args.batch_size, shuffle=True
    )
    amazon_test_loader = torch.utils.data.DataLoader(
        amazon_testset, batch_size=args.batch_size, shuffle=False
    )

    caltech_train_loader = torch.utils.data.DataLoader(
        caltech_trainset, batch_size=args.batch_size, shuffle=True
    )
    caltech_test_loader = torch.utils.data.DataLoader(
        caltech_testset, batch_size=args.batch_size, shuffle=False
    )

    dslr_train_loader = torch.utils.data.DataLoader(
        dslr_trainset, batch_size=args.batch_size, shuffle=True
    )
    dslr_test_loader = torch.utils.data.DataLoader(
        dslr_testset, batch_size=args.batch_size, shuffle=False
    )

    webcam_train_loader = torch.utils.data.DataLoader(
        webcam_trainset, batch_size=args.batch_size, shuffle=True
    )
    webcam_test_loader = torch.utils.data.DataLoader(
        webcam_testset, batch_size=args.batch_size, shuffle=False
    )

    test_loaders = [
        amazon_test_loader,
        caltech_test_loader,
        dslr_test_loader,
        webcam_test_loader,
    ]
    train_sets = [
        amazon_trainset,
        caltech_trainset,
        dslr_trainset,
        webcam_trainset,
    ]

    train_loaders = [
        amazon_train_loader,
        caltech_train_loader,
        dslr_train_loader,
        webcam_train_loader,
    ]
    return train_loaders, train_sets, test_loaders


def load_svhn_data(datadir):

    transform = transforms.Compose([transforms.ToTensor()])

    svhn_train_ds = SVHN_custom(datadir, train=True, download=True, transform=transform)
    svhn_test_ds = SVHN_custom(datadir, train=False, download=True, transform=transform)

    X_train, y_train = svhn_train_ds.data, svhn_train_ds.target
    X_test, y_test = svhn_test_ds.data, svhn_test_ds.target

    # X_train = X_train.data.numpy()
    # y_train = y_train.data.numpy()
    # X_test = X_test.data.numpy()
    # y_test = y_test.data.numpy()

    return (X_train, y_train, X_test, y_test)


def load_cifar10_data(datadir):

    transform = transforms.Compose([transforms.ToTensor()])

    cifar10_train_ds = CIFAR10_truncated(
        datadir, train=True, download=True, transform=transform
    )
    cifar10_test_ds = CIFAR10_truncated(
        datadir, train=False, download=True, transform=transform
    )

    X_train, y_train = cifar10_train_ds.data, cifar10_train_ds.target
    X_test, y_test = cifar10_test_ds.data, cifar10_test_ds.target

    # y_train = y_train.numpy()
    # y_test = y_test.numpy()

    return (X_train, y_train, X_test, y_test)


def load_celeba_data(datadir):

    transform = transforms.Compose([transforms.ToTensor()])

    celeba_train_ds = CelebA_custom(
        datadir, split="train", target_type="attr", download=True, transform=transform
    )
    celeba_test_ds = CelebA_custom(
        datadir, split="test", target_type="attr", download=True, transform=transform
    )

    gender_index = celeba_train_ds.attr_names.index("Male")
    y_train = celeba_train_ds.attr[:, gender_index : gender_index + 1].reshape(-1)
    y_test = celeba_test_ds.attr[:, gender_index : gender_index + 1].reshape(-1)

    # y_train = y_train.numpy()
    # y_test = y_test.numpy()

    return (None, y_train, None, y_test)


def load_femnist_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])

    mnist_train_ds = FEMNIST(datadir, train=True, transform=transform, download=True)
    mnist_test_ds = FEMNIST(datadir, train=False, transform=transform, download=True)

    X_train, y_train, u_train = (
        mnist_train_ds.data,
        mnist_train_ds.targets,
        mnist_train_ds.users_index,
    )
    X_test, y_test, u_test = (
        mnist_test_ds.data,
        mnist_test_ds.targets,
        mnist_test_ds.users_index,
    )

    X_train = X_train.data.numpy()
    y_train = y_train.data.numpy()
    u_train = np.array(u_train)
    X_test = X_test.data.numpy()
    y_test = y_test.data.numpy()
    u_test = np.array(u_test)

    return (X_train, y_train, u_train, X_test, y_test, u_test)


def load_cifar100_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])

    cifar100_train_ds = CIFAR100_truncated(
        datadir, train=True, download=True, transform=transform
    )
    cifar100_test_ds = CIFAR100_truncated(
        datadir, train=False, download=True, transform=transform
    )

    X_train, y_train = cifar100_train_ds.data, cifar100_train_ds.target
    X_test, y_test = cifar100_test_ds.data, cifar100_test_ds.target

    # y_train = y_train.numpy()
    # y_test = y_test.numpy()

    return (X_train, y_train, X_test, y_test)


def record_net_data_stats(y_train, net_dataidx_map, logdir):

    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp

    logger.info("Data statistics: %s" % str(net_cls_counts))

    return net_cls_counts


def load_tinyimagenet_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])
    xray_train_ds = ImageFolder_custom(datadir + "./train/", transform=transform)
    xray_test_ds = ImageFolder_custom(datadir + "./val/", transform=transform)

    X_train, y_train = np.array([s[0] for s in xray_train_ds.samples]), np.array(
        [int(s[1]) for s in xray_train_ds.samples]
    )
    X_test, y_test = np.array([s[0] for s in xray_test_ds.samples]), np.array(
        [int(s[1]) for s in xray_test_ds.samples]
    )

    return (X_train, y_train, X_test, y_test)


def partition_data(dataset, datadir, logdir, partition, n_parties, beta=0.4):
    # np.random.seed(2020)
    # torch.manual_seed(2020)

    if dataset == "mnist":
        X_train, y_train, X_test, y_test = load_mnist_data(datadir)
    elif dataset == "fmnist":
        X_train, y_train, X_test, y_test = load_fmnist_data(datadir)
    elif dataset == "cifar10":
        X_train, y_train, X_test, y_test = load_cifar10_data(datadir)
    elif dataset.lower() == "svhn":
        X_train, y_train, X_test, y_test = load_svhn_data(datadir)
    elif dataset == "celeba":
        X_train, y_train, X_test, y_test = load_celeba_data(datadir)
    elif dataset == "femnist":
        X_train, y_train, u_train, X_test, y_test, u_test = load_femnist_data(datadir)
    elif dataset == "cifar100":
        X_train, y_train, X_test, y_test = load_cifar100_data(datadir)
    elif dataset == "tinyimagenet":
        X_train, y_train, X_test, y_test = load_tinyimagenet_data(datadir)
    elif dataset == "generated":
        X_train, y_train = [], []
        for loc in range(4):
            for i in range(1000):
                p1 = random.random()
                p2 = random.random()
                p3 = random.random()
                if loc > 1:
                    p2 = -p2
                if loc % 2 == 1:
                    p3 = -p3
                if i % 2 == 0:
                    X_train.append([p1, p2, p3])
                    y_train.append(0)
                else:
                    X_train.append([-p1, -p2, -p3])
                    y_train.append(1)
        X_test, y_test = [], []
        for i in range(1000):
            p1 = random.random() * 2 - 1
            p2 = random.random() * 2 - 1
            p3 = random.random() * 2 - 1
            X_test.append([p1, p2, p3])
            if p1 > 0:
                y_test.append(0)
            else:
                y_test.append(1)
        X_train = np.array(X_train, dtype=np.float32)
        X_test = np.array(X_test, dtype=np.float32)
        y_train = np.array(y_train, dtype=np.int32)
        y_test = np.array(y_test, dtype=np.int64)
        idxs = np.linspace(0, 3999, 4000, dtype=np.int64)
        batch_idxs = np.array_split(idxs, n_parties)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_parties)}
        mkdirs("data/generated/")
        np.save("data/generated/X_train.npy", X_train)
        np.save("data/generated/X_test.npy", X_test)
        np.save("data/generated/y_train.npy", y_train)
        np.save("data/generated/y_test.npy", y_test)

    # elif dataset == 'covtype':
    #    cov_type = sk.fetch_covtype('./data')
    #    num_train = int(581012 * 0.75)
    #    idxs = np.random.permutation(581012)
    #    X_train = np.array(cov_type['data'][idxs[:num_train]], dtype=np.float32)
    #    y_train = np.array(cov_type['target'][idxs[:num_train]], dtype=np.int32) - 1
    #    X_test = np.array(cov_type['data'][idxs[num_train:]], dtype=np.float32)
    #    y_test = np.array(cov_type['target'][idxs[num_train:]], dtype=np.int32) - 1
    #    mkdirs("data/generated/")
    #    np.save("data/generated/X_train.npy",X_train)
    #    np.save("data/generated/X_test.npy",X_test)
    #    np.save("data/generated/y_train.npy",y_train)
    #    np.save("data/generated/y_test.npy",y_test)

    elif dataset in ("rcv1", "SUSY", "covtype"):
        X_train, y_train = load_svmlight_file(datadir + dataset)
        X_train = X_train.todense()
        num_train = int(X_train.shape[0] * 0.75)
        if dataset == "covtype":
            y_train = y_train - 1
        else:
            y_train = (y_train + 1) / 2
        idxs = np.random.permutation(X_train.shape[0])

        X_test = np.array(X_train[idxs[num_train:]], dtype=np.float32)
        y_test = np.array(y_train[idxs[num_train:]], dtype=np.int32)
        X_train = np.array(X_train[idxs[:num_train]], dtype=np.float32)
        y_train = np.array(y_train[idxs[:num_train]], dtype=np.int32)

        mkdirs("data/generated/")
        np.save("data/generated/X_train.npy", X_train)
        np.save("data/generated/X_test.npy", X_test)
        np.save("data/generated/y_train.npy", y_train)
        np.save("data/generated/y_test.npy", y_test)

    elif dataset in ("a9a"):
        X_train, y_train = load_svmlight_file(datadir + "a9a")
        X_test, y_test = load_svmlight_file(datadir + "a9a.t")
        X_train = X_train.todense()
        X_test = X_test.todense()
        X_test = np.c_[
            X_test, np.zeros((len(y_test), X_train.shape[1] - np.size(X_test[0, :])))
        ]

        X_train = np.array(X_train, dtype=np.float32)
        X_test = np.array(X_test, dtype=np.float32)
        y_train = (y_train + 1) / 2
        y_test = (y_test + 1) / 2
        y_train = np.array(y_train, dtype=np.int32)
        y_test = np.array(y_test, dtype=np.int32)

        mkdirs("data/generated/")
        np.save("data/generated/X_train.npy", X_train)
        np.save("data/generated/X_test.npy", X_test)
        np.save("data/generated/y_train.npy", y_train)
        np.save("data/generated/y_test.npy", y_test)

    n_train = y_train.shape[0]

    if partition == "homo":
        idxs = np.random.permutation(n_train)
        batch_idxs = np.array_split(idxs, n_parties)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_parties)}

    elif partition == "noniid-labeldir":
        min_size = 0
        min_require_size = 10
        K = 10
        preallocate = 10  # Set the number of samples to preallocate
        if dataset in ("celeba", "covtype", "a9a", "rcv1", "SUSY"):
            K = 2
        if dataset == "cifar100":
            K = 100
        elif dataset == "tinyimagenet":
            K = 200
        N = y_train.shape[0]
        net_dataidx_map = {}
        while min_size < min_require_size:
            idx_batch = [[] for _ in range(n_parties)]
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                # Ensure each party gets at least preallocate samples of class k
                for i in range(n_parties):
                    if len(idx_k) > preallocate - 1:
                        idx_batch[i].extend(idx_k[:preallocate])
                        idx_k = idx_k[preallocate:]
                proportions = np.random.dirichlet(np.repeat(beta, n_parties))
                proportions = np.array(
                    [
                        p * (len(idx_j) < N / n_parties)
                        for p, idx_j in zip(proportions, idx_batch)
                    ]
                )
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [
                    idx_j + idx.tolist()
                    for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))
                ]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(n_parties):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

    elif partition > "noniid-#label0" and partition <= "noniid-#label9":
        num = eval(partition[13:])
        if dataset in ("celeba", "covtype", "a9a", "rcv1", "SUSY"):
            num = 1
            K = 2
        else:
            K = 10
        if dataset == "cifar100":
            K = 100
        elif dataset == "tinyimagenet":
            K = 200
        if num == 10:
            net_dataidx_map = {
                i: np.ndarray(0, dtype=np.int64) for i in range(n_parties)
            }
            for i in range(10):
                idx_k = np.where(y_train == i)[0]
                np.random.shuffle(idx_k)
                split = np.array_split(idx_k, n_parties)
                for j in range(n_parties):
                    net_dataidx_map[j] = np.append(net_dataidx_map[j], split[j])
        else:
            times = [0 for i in range(K)]
            contain = []
            for i in range(n_parties):
                current = [i % K]
                times[i % K] += 1
                j = 1
                while j < num:
                    ind = random.randint(0, K - 1)
                    if ind not in current:
                        j = j + 1
                        current.append(ind)
                        times[ind] += 1
                contain.append(current)
            net_dataidx_map = {
                i: np.ndarray(0, dtype=np.int64) for i in range(n_parties)
            }
            for i in range(K):
                idx_k = np.where(y_train == i)[0]
                np.random.shuffle(idx_k)
                split = np.array_split(idx_k, times[i])
                ids = 0
                for j in range(n_parties):
                    if i in contain[j]:
                        net_dataidx_map[j] = np.append(net_dataidx_map[j], split[ids])
                        ids += 1

    elif partition == "iid-diff-quantity":
        idxs = np.random.permutation(n_train)
        min_size = 0
        while min_size < 10:
            proportions = np.random.dirichlet(np.repeat(beta, n_parties))
            proportions = proportions / proportions.sum()
            min_size = np.min(proportions * len(idxs))
        proportions = (np.cumsum(proportions) * len(idxs)).astype(int)[:-1]
        batch_idxs = np.split(idxs, proportions)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_parties)}

    elif partition == "mixed":
        min_size = 0
        min_require_size = 10
        K = 10
        if dataset in ("celeba", "covtype", "a9a", "rcv1", "SUSY"):
            K = 2
            # min_require_size = 100

        N = y_train.shape[0]
        net_dataidx_map = {}

        times = [1 for i in range(10)]
        contain = []
        for i in range(n_parties):
            current = [i % K]
            j = 1
            while j < 2:
                ind = random.randint(0, K - 1)
                if ind not in current and times[ind] < 2:
                    j = j + 1
                    current.append(ind)
                    times[ind] += 1
            contain.append(current)
        net_dataidx_map = {i: np.ndarray(0, dtype=np.int64) for i in range(n_parties)}

        min_size = 0
        while min_size < 10:
            proportions = np.random.dirichlet(np.repeat(beta, n_parties))
            proportions = proportions / proportions.sum()
            min_size = np.min(proportions * n_train)

        for i in range(K):
            idx_k = np.where(y_train == i)[0]
            np.random.shuffle(idx_k)

            proportions_k = np.random.dirichlet(np.repeat(beta, 2))
            # proportions_k = np.ndarray(0,dtype=np.float64)
            # for j in range(n_parties):
            #    if i in contain[j]:
            #        proportions_k=np.append(proportions_k ,proportions[j])

            proportions_k = (np.cumsum(proportions_k) * len(idx_k)).astype(int)[:-1]

            split = np.split(idx_k, proportions_k)
            ids = 0
            for j in range(n_parties):
                if i in contain[j]:
                    net_dataidx_map[j] = np.append(net_dataidx_map[j], split[ids])
                    ids += 1

    elif partition == "real" and dataset == "femnist":
        num_user = u_train.shape[0]
        user = np.zeros(num_user + 1, dtype=np.int32)
        for i in range(1, num_user + 1):
            user[i] = user[i - 1] + u_train[i - 1]
        no = np.random.permutation(num_user)
        batch_idxs = np.array_split(no, n_parties)
        net_dataidx_map = {i: np.zeros(0, dtype=np.int32) for i in range(n_parties)}
        for i in range(n_parties):
            for j in batch_idxs[i]:
                net_dataidx_map[i] = np.append(
                    net_dataidx_map[i], np.arange(user[j], user[j + 1])
                )

    elif partition == "transfer-from-femnist":
        stat = np.load("femnist-dis.npy")
        n_total = stat.shape[0]
        chosen = np.random.permutation(n_total)[:n_parties]
        stat = stat[chosen, :]

        if dataset in ("celeba", "covtype", "a9a", "rcv1", "SUSY"):
            K = 2
        else:
            K = 10

        N = y_train.shape[0]
        # np.random.seed(2020)
        net_dataidx_map = {}

        idx_batch = [[] for _ in range(n_parties)]
        for k in range(K):
            idx_k = np.where(y_train == k)[0]
            np.random.shuffle(idx_k)
            proportions = stat[:, k]
            # logger.info("proportions2: ", proportions)
            proportions = proportions / proportions.sum()
            # logger.info("proportions3: ", proportions)
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            # logger.info("proportions4: ", proportions)
            idx_batch = [
                idx_j + idx.tolist()
                for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))
            ]

        for j in range(n_parties):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

    elif partition == "transfer-from-criteo":
        stat0 = np.load("criteo-dis.npy")

        n_total = stat0.shape[0]
        flag = True
        while flag:
            chosen = np.random.permutation(n_total)[:n_parties]
            stat = stat0[chosen, :]
            check = [0 for i in range(10)]
            for ele in stat:
                for j in range(10):
                    if ele[j] > 0:
                        check[j] = 1
            flag = False
            for i in range(10):
                if check[i] == 0:
                    flag = True
                    break

        if dataset in ("celeba", "covtype", "a9a", "rcv1", "SUSY"):
            K = 2
            stat[:, 0] = np.sum(stat[:, :5], axis=1)
            stat[:, 1] = np.sum(stat[:, 5:], axis=1)
        else:
            K = 10

        N = y_train.shape[0]
        # np.random.seed(2020)
        net_dataidx_map = {}

        idx_batch = [[] for _ in range(n_parties)]
        for k in range(K):
            idx_k = np.where(y_train == k)[0]
            np.random.shuffle(idx_k)
            proportions = stat[:, k]
            # logger.info("proportions2: ", proportions)
            proportions = proportions / proportions.sum()
            # logger.info("proportions3: ", proportions)
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            # logger.info("proportions4: ", proportions)
            idx_batch = [
                idx_j + idx.tolist()
                for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))
            ]

        for j in range(n_parties):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

    traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map, logdir)
    return (X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts)


class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=1.0):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + "(mean={0}, std={1})".format(
            self.mean, self.std
        )


def get_dataloader(
    args,
    data_np,
    label_np,
    dataset_type,
    datadir,
    train_bs,
    is_labeled=None,
    data_idxs=None,
    is_testing=False,
    pre_sz=40,
    input_sz=32,
    noise_level=0,
):
    if dataset_type == "SVHN":
        normalize = transforms.Normalize(
            mean=[0.4376821, 0.4437697, 0.47280442],
            std=[0.19803012, 0.20101562, 0.19703614],
        )
        assert (
            pre_sz == 40 and input_sz == 32
        ), "Error: Wrong input size for 32*32 dataset"
    elif dataset_type == "cifar10":
        normalize = transforms.Normalize(
            mean=[0.49139968, 0.48215827, 0.44653124],
            std=[0.24703233, 0.24348505, 0.26158768],
        )
        assert (
            pre_sz == 40 and input_sz == 32
        ), "Error: Wrong input size for 32*32 dataset"
    elif dataset_type == "cifar100":
        normalize = transforms.Normalize(
            mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
            std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404],
        )
        assert (
            pre_sz == 40 and input_sz == 32
        ), "Error: Wrong input size for 32*32 dataset"
    elif dataset_type == "skin":
        normalize = transforms.Normalize(
            mean=[0.7630332, 0.5456457, 0.57004654],
            std=[0.14092809, 0.15261231, 0.16997086],
        )
    elif dataset_type == "generated":
        # ds_gene = dataset.Generated(datadir, data_idxs, train=True)
        # dl_gene = data.DataLoader(dataset=ds_gene, batch_size=train_bs, drop_last=False, shuffle=True, num_workers=8)
        pass
    elif dataset_type == "femnist":
        normalize = (None,)
        assert (
            pre_sz == 36 and input_sz == 32
        ), "Error: Wrong input size for 32*32 dataset"
    if not is_testing:
        if is_labeled:
            trans = transforms.Compose(
                [
                    transforms.RandomCrop(size=(input_sz, input_sz)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ToTensor(),
                    AddGaussianNoise(0.0, noise_level),
                    normalize,
                ]
            )
            ds = CheXpertDataset(
                dataset_type,
                data_np,
                label_np,
                pre_sz,
                pre_sz,
                lab_trans=trans,
                is_labeled=True,
                is_testing=False,
            )
        else:
            weak_trans = transforms.Compose(
                [
                    transforms.RandomCrop(size=(input_sz, input_sz)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ToTensor(),
                    AddGaussianNoise(0.0, noise_level),
                    normalize,
                ]
            )
            strong_trans = transforms.Compose(
                [
                    transforms.RandomResizedCrop(
                        size=(input_sz, input_sz)
                    ),  # 更大范围的裁剪
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomApply(
                        [
                            transforms.ColorJitter(
                                0.4, 0.4, 0.4, 0.1
                            )  # 随机调整亮度、对比度、饱和度和色调
                        ],
                        p=0.8,
                    ),  # 以较高概率应用颜色抖动
                    transforms.RandomGrayscale(p=0.2),  # 20% 概率将图像转为灰度
                    transforms.ToTensor(),
                    AddGaussianNoise(0.0, noise_level),  # 加入高斯噪声
                    normalize,
                ]
            )

            ds = CheXpertDataset(
                dataset_type,
                data_np,
                label_np,
                pre_sz,
                pre_sz,
                un_trans_wk=TransformTwice(weak_trans, strong_trans),
                data_idxs=data_idxs,
                is_labeled=False,
                is_testing=False,
            )
        dl = data.DataLoader(
            dataset=ds,
            batch_size=train_bs,
            drop_last=False,
            shuffle=True,
            num_workers=8,
        )
    else:
        if dataset_type == "generated":
            pass
        else:
            ds = CheXpertDataset(
                dataset_type,
                data_np,
                label_np,
                input_sz,
                input_sz,
                lab_trans=transforms.Compose(
                    [
                        # K.RandomCrop((224, 224)),
                        transforms.ToTensor(),
                        normalize,
                    ]
                ),
                is_labeled=True,
                is_testing=True,
            )
            dl = data.DataLoader(
                dataset=ds,
                batch_size=train_bs,
                drop_last=False,
                shuffle=False,
                num_workers=8,
            )
    return dl, ds


def update_ema_variables(model, ema_model, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)


def label_guessing(
    model: nn.Module, batches_1: Sequence[Tensor], model_type=None
) -> Tensor:
    model.eval()
    with torch.no_grad():
        feature = [model(batch, model=model_type)[1] for batch in batches_1]
        probs = [
            F.softmax(model(batch, model=model_type)[2], dim=1) for batch in batches_1
        ]
        mean_feature = reduce(lambda x, y: x + y, feature) / len(batches_1)
        mean_prob = reduce(lambda x, y: x + y, probs) / len(batches_1)

    return mean_feature, mean_prob


# def label_guessing(model: nn.Module, batches_1: Sequence[Tensor], model_type=None) -> Tensor:
#     model.eval()
#     with torch.no_grad():
#         probs = [F.softmax(model(batch, model=model_type)[2], dim=1)
#                  for batch in batches_1]
#         mean_prob = reduce(lambda x, y: x + y, probs) / len(batches_1)

#     return mean_prob


def sharpen(x: Tensor, t=0.5) -> Tensor:
    sharpened_x = x ** (1 / t)
    return sharpened_x / sharpened_x.sum(dim=1, keepdim=True)


# 实现一个warmming-up alpha 计算函数，基于线性增长，传入参数为warmup 的最大round, 这个线性增长 alpha_t = alpha_0 + t / (alpha_n - alpha_0) * t / rounds_max,每次返回一个值
class RampUp(ABC):
    def __init__(self, length: int, current: int = 0):
        self.current = current
        self.length = length

    @abstractmethod
    def __call__(self, current: Optional[int] = None, is_step: bool = True) -> float:
        pass

    def state_dict(self) -> Dict[str, Any]:
        return dict(current=self.current, length=self.length)

    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True):
        if strict:
            is_equal, incompatible_keys = self._verify_state_dict(state_dict)
            assert (
                is_equal
            ), f"loaded state dict contains incompatible keys: {incompatible_keys}"

        # for attr_name, attr_value in state_dict.items():
        #     if attr_name in self.__dict__:
        #         self.__dict__[attr_name] = attr_value

        self.current = state_dict["current"]
        self.length = state_dict["length"]

    def _verify_state_dict(self, state_dict: Dict[str, Any]) -> Tuple[bool, Set[str]]:
        self_keys = set(self.__dict__.keys())
        state_dict_keys = set(state_dict.keys())
        incompatible_keys = self_keys.union(state_dict_keys) - self_keys.intersection(
            state_dict_keys
        )
        is_equal = len(incompatible_keys) == 0

        return is_equal, incompatible_keys

    def _update_step(self, is_step: bool):
        if is_step:
            self.current += 1


class LinearRampUp(RampUp):
    def __init__(self, length: int, alpha_0: float, alpha_t: float):
        super(LinearRampUp, self).__init__(length)
        self.alpha_0 = alpha_0
        self.alpha_t = alpha_t

    def __call__(self, current: Optional[int] = None, is_step: bool = True) -> float:
        if current is not None:
            self.current = current

        if self.current >= self.length:
            ramp_up = self.alpha_t
        else:
            ramp_up = (
                self.alpha_0
                + (self.alpha_t - self.alpha_0) * self.current / self.length
            )

        self._update_step(is_step)

        return ramp_up


def softmax_mse_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    # input_softmax = F.softmax(input_logits, dim=1)
    # target_softmax = F.softmax(target_logits, dim=1)

    mse_loss = (input_logits - target_logits) ** 2
    # print(mse_loss)
    return mse_loss


class WPOptim(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, alpha=0.05, **kwargs):
        defaults = dict(alpha=alpha, **kwargs)
        super(WPOptim, self).__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def generate_delta(self, zero_grad=False):
        device = self.param_groups[0]["params"][0].device
        grad_norm = torch.norm(
            torch.stack(
                [
                    (1.0 * p.grad).norm(p=2).to(device)
                    for group in self.param_groups
                    for p in group["params"]
                    if p.grad is not None
                ]
            ),
            p=2,
        )
        for group in self.param_groups:
            scale = group["alpha"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                delta = 1.0 * p.grad * scale.to(p)
                p.add_(delta)
                self.state[p]["delta"] = delta

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.sub_(self.state[p]["delta"])
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()


# 定义 InfoNCE Loss 函数
def info_nce_loss(features1, features2, masks=None, temperature=0.07):
    # 动态获取批次大小
    batch_size = features1.size(0)

    # 将两个视角的特征拼接在一起，形状变为 (2B, D)
    features = torch.cat([features1, features2], dim=0)  # [2B, D]
    # print(f"features 的形状: {features.shape}")  # [2B, D]

    # 生成标签
    labels = torch.cat([torch.arange(batch_size) for _ in range(2)], dim=0)  # [2B,]
    labels = labels.unsqueeze(0) == labels.unsqueeze(1)  # [2B, 2B]
    labels = labels.float().to(features.device)

    # 归一化特征
    features = F.normalize(features, dim=1)

    # 计算相似度矩阵
    similarity_matrix = torch.matmul(features, features.T)  # [2B, 2B]
    # print(f"similarity_matrix 的形状: {similarity_matrix.shape}")  # [2B, 2B]

    # 创建掩码，去除对角线（即每个样本与自身的相似度）
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(features.device)
    labels = labels[~mask].view(labels.shape[0], -1)  # [2B, 2B-1]
    similarity_matrix = similarity_matrix[~mask].view(
        similarity_matrix.shape[0], -1
    )  # [2B, 2B-1]

    # print(f"labels 的形状: {labels.shape}")  # [2B, 2B-1]
    # print(f"similarity_matrix 的形状: {similarity_matrix.shape}")  # [2B, 2B-1]

    # 选择正样本和负样本
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)  # [2B, 1]
    negatives = similarity_matrix[~labels.bool()].view(
        similarity_matrix.shape[0], -1
    )  # [2B, 2B-2]

    # 拼接正样本和负样本
    logits = torch.cat([positives, negatives], dim=1)  # [2B, 2B-1]
    # print(f"logits 的形状: {logits.shape}")  # [2B, 2B-1]

    # 创建标签，正样本的位置为0
    ce_labels = torch.zeros(logits.shape[0], dtype=torch.long).to(
        features.device
    )  # [2B,]

    # 缩放 logits
    logits = logits / temperature

    if masks is not None:
        # print(f"masks 的形状: {masks.shape}")  # [2B,]
        if masks.dim() == 1 and masks.size(0) == logits.size(0):
            # 使用加权交叉熵损失
            loss = F.cross_entropy(logits, ce_labels, reduction="none")  # [2B,]
            loss = loss * masks  # [2B,]
            loss = loss.mean()
            # print(f"加权后的损失: {loss.item()}")
        else:
            raise ValueError(
                f"masks 的形状必须为 ({logits.size(0)},)，但得到 {masks.shape}"
            )
    else:
        # 计算标准交叉熵损失
        loss = F.cross_entropy(logits, ce_labels)
        # print(f"标准交叉熵损失: {loss.item()}")

    return loss
