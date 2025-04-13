from validation import epochVal_metrics_test
from options import args_parser
import os
import sys
import logging
import random
import numpy as np
import copy
import datetime
from Server import FedAvg
import torch
import torch.optim as optim
from torchvision import transforms
import torch.backends.cudnn as cudnn
from models import ModelFedCon

# from dataloaders import dataset
from utils import partition_data, get_dataloader, update_ema_variables, WPOptim, prepare_data
from labeled_client import SupervisedLocalUpdate
from unlabelled_client import UnsupervisedLocalUpdate
from tqdm import tqdm, trange
from layers import AmpNorm

# from cifar_load import get_dataloader, partition_data, partition_data_allnoniid
from torch.utils.tensorboard import SummaryWriter


def test(
    epoch, checkpoint, data_test=None, label_test=None, n_classes=10, test_loaders=None
):
    net = ModelFedCon(args.input_shape, args.model, args.out_dim, n_classes=n_classes)
    if len(args.gpu.split(",")) > 1:
        net = torch.nn.DataParallel(
            net, device_ids=[i for i in range(round(len(args.gpu) / 2))]
        )
    model = net.cuda()
    model.load_state_dict(checkpoint)

    if (
        args.dataset == "SVHN"
        or args.dataset == "cifar100"
        or args.dataset == "cifar10"
    ):
        test_dl, test_ds = get_dataloader(
            args,
            data_test,
            label_test,
            args.dataset,
            args.datadir,
            args.batch_size,
            is_labeled=True,
            is_testing=True,
        )
    elif args.dataset == "skin" or args.dataset == "femnist":
        test_dl, test_ds = get_dataloader(
            args,
            data_test,
            label_test,
            args.dataset,
            args.datadir,
            args.batch_size,
            is_labeled=True,
            is_testing=True,
            pre_sz=args.pre_sz,
            input_sz=args.input_sz,
        )
    test_dls = test_loaders
    # 兼容单个 test_loader 的原有逻辑
    if test_dls is None and test_dl is not None:
        AUROCs, Accus, Pre, Recall = epochVal_metrics_test(
            model, test_dl, args.model, n_classes=n_classes
        )
        AUROC_avg = np.array(AUROCs).mean()
        Accus_avg = np.array(Accus).mean()
        Pre_avg = np.array(Pre).mean()
        Recall_avg = np.array(Recall).mean()
        return AUROC_avg, Accus_avg, Pre_avg, Recall_avg

    # 处理多个 test_loaders 的情况
    elif test_dls is not None and isinstance(test_dls, list):
        all_AUROCs = []
        all_Accus = []
        all_Pres = []
        all_Recalls = []

        for idx, loader in enumerate(test_dls):
            print(f"Evaluating test_loader {idx + 1}/{len(test_dls)}...")
            AUROCs, Accus, Pre, Recall = epochVal_metrics_test(
                model, loader, args.model, n_classes=n_classes
            )
            all_AUROCs.append(np.array(AUROCs).mean())
            all_Accus.append(Accus)
            all_Pres.append(Pre)
            all_Recalls.append(Recall)

        AUROC_avg = np.mean(all_AUROCs)
        Accus_avg = np.mean(all_Accus)
        Pre_avg = np.mean(all_Pres)
        Recall_avg = np.mean(all_Recalls)

        return AUROC_avg, Accus_avg

    else:
        raise ValueError("Either test_loader or test_loaders must be provided.")


if __name__ == "__main__":
    args = args_parser()
    # 设置客户端比例
    # supervised_user_id = [0,1,2,3,4,5,6,7,8,9]
    supervised_user_id = [0]
    # supervised_user_id = [0]
    unsupervised_user_id = list(
        range(len(supervised_user_id), args.unsup_num + len(supervised_user_id))
    )
    sup_num = len(supervised_user_id)
    unsup_num = len(unsupervised_user_id)
    total_num = sup_num + unsup_num

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    time_current = "rc_1_office-6"

    # 日志记录
    # if args.log_file_name is None:
    #     args.log_file_name = 'log-%s' % (datetime.datetime.now().strftime("%m-%d-%H%M-%S"))
    # log_path = args.log_file_name + '.log'
    # os.makedirs(args.logdir, exist_ok=True)
    # logging.basicConfig(filename=os.path.join(args.logdir, log_path), level=logging.INFO,
    #                     format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    # logger = logging.getLogger()
    # logger.info(str(args))
    # logger.info(time_current)

    # 设置日志文件名
    if args.log_file_name is None:
        args.log_file_name = "%s-%s" % (
            time_current,
            datetime.datetime.now().strftime("%m-%d-%H%M-%S"),
        )
    log_path = args.log_file_name + ".log"

    # 确保日志目录存在
    os.makedirs(args.logdir, exist_ok=True)

    # 配置日志
    log_file = os.path.join(args.logdir, log_path)
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d] %(message)s",
        datefmt="%H:%M:%S",
    )

    # 创建文件处理器并添加到日志记录器
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("[%(asctime)s.%(msecs)03d] %(message)s", datefmt="%H:%M:%S")
    )

    # 获取默认的日志记录器并添加文件处理器
    logger = logging.getLogger()
    logger.addHandler(file_handler)

    # 确保之前的处理器不会干扰
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler):
            logger.removeHandler(handler)

    # 测试日志记录
    logger.info(str(args))
    logger.info(time_current)
    logger.info("This is a test log message.")

    print("Logging configuration complete. Logs should be in:", log_file)

    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    # tensorboard设置
    if not os.path.isdir("tensorboard"):
        os.mkdir("tensorboard")
    if args.dataset == "SVHN":
        if not os.path.isdir("tensorboard/SVHN/" + time_current):
            os.mkdir("tensorboard/SVHN/" + time_current)
        writer = SummaryWriter("tensorboard/SVHN/" + time_current)

    if args.dataset == "cifar10":
        if not os.path.isdir("tensorboard/cifar10/" + time_current):
            os.mkdir("tensorboard/cifar10/" + time_current)
        writer = SummaryWriter("tensorboard/cifar10/" + time_current)

    if args.dataset == "office_caltech":
        if not os.path.isdir("tensorboard/office_caltech/" + time_current):
            os.mkdir("tensorboard/office_caltech/" + time_current)
        writer = SummaryWriter("tensorboard/office_caltech/" + time_current)

    if args.dataset == "feminst":
        if not os.path.isdir("tensorboard/femnisst/" + time_current):
            os.mkdir("tensorboard/femnist/" + time_current)
        writer = SummaryWriter("tensorboard/femnist/" + time_current)

    elif args.dataset == "cifar100":
        if not os.path.isdir("tensorboard/cifar100/" + time_current):
            os.mkdir("tensorboard/cifar100/" + time_current)
        writer = SummaryWriter("tensorboard/cifar100/" + time_current)

    elif args.dataset == "skin":
        if not os.path.isdir("tensorboard/skin/" + time_current):
            os.mkdir("tensorboard/skin/" + time_current)
        writer = SummaryWriter("tensorboard/skin/" + time_current)

    snapshot_path = "model/"
    if not os.path.isdir(snapshot_path):
        os.mkdir(snapshot_path)
    if args.dataset == "SVHN":
        snapshot_path = "model/SVHN/"
    if args.dataset == "cifar10":
        snapshot_path = "model/cifar10/"
    if args.dataset == "cifar100":
        snapshot_path = "model/cifar100/"
    if args.dataset == "skin":
        snapshot_path = "model/skin/"
    if args.dataset == "femnist":
        snapshot_path = "model/femnist/"
    if args.dataset == "office_caltech":
        snapshot_path = "model/office_caltech/"
    if not os.path.isdir(snapshot_path):
        os.mkdir(snapshot_path)

    if args.dataset == "office_caltech":
        assert args.n_parties == 4, "Error: Wrong client number"
        train_loaders, train_sets, test_loaders = prepare_data(args)
    # 划分数据集
    else:
        X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = (
            partition_data(
                args.dataset,
                args.datadir,
                args.logdir,
                args.partition,
                args.n_parties,
                beta=args.beta,
            )
        )

    if args.dataset == "SVHN":
        X_train = X_train.transpose([0, 2, 3, 1])
        X_test = X_test.transpose([0, 2, 3, 1])

    # 初始化全局模型
    args.input_shape = (3, 32, 32)

    if args.dataset == "office_caltech":
        n_classes = 10
    else:
        n_classes = len(np.unique(y_train))

    # 初始化全局模型
    net_glob = ModelFedCon(
        args.input_shape, args.model, args.out_dim, n_classes=n_classes
    )
    tea_glob = ModelFedCon(
        args.input_shape, args.model, args.out_dim, n_classes=n_classes
    )
    if args.resume:
        print("==> Resuming from checkpoint..")
        if args.dataset == "cifar100":
            checkpoint = torch.load("warmup/cifar100.pth")
        elif args.dataset == "SVHN":
            checkpoint = torch.load("warmup/SVHN.pth")

        net_glob.load_state_dict(checkpoint["state_dict"])
        start_epoch = 7
    else:
        start_epoch = 0

    if len(args.gpu.split(",")) > 1:
        net_glob = torch.nn.DataParallel(
            net_glob, device_ids=[i for i in range(round(len(args.gpu) / 2))]
        )

    net_glob.train()
    tea_glob.eval()
    weight_glob = copy.deepcopy(net_glob.state_dict())

    # 初始化本地模型
    # labeled clients list
    # l_weight_locals = []
    l_trainer_locals = []
    l_net_locals = []
    l_optim_locals = []

    # 初始化监督学习客户端
    for i in supervised_user_id:
        l_trainer_locals.append(
            SupervisedLocalUpdate(args, n_classes)
        )
        # l_weight_locals.append(copy.deepcopy(weight_glob))
        l_net_locals.append(copy.deepcopy(net_glob))
        if args.opt == "adam":
            optimizer = torch.optim.Adam(
                l_net_locals[i].parameters(),
                lr=args.base_lr,
                betas=(0.9, 0.999),
                weight_decay=5e-4,
            )
        elif args.opt == "sgd":
            optimizer = torch.optim.SGD(
                l_net_locals[i].parameters(),
                lr=args.base_lr,
                momentum=0.9,
                weight_decay=5e-4,
            )
        elif args.opt == "adamw":
            optimizer = torch.optim.AdamW(
                l_net_locals[i].parameters(), lr=args.base_lr, weight_decay=0.02
            )
        elif args.opt == "wpoptim":
            # optimizer = torch.optim.SGD(l_net_locals[i].parameters(), lr=args.base_lr, momentum=0.9,
            #                             weight_decay=5e-4)
            optimizer = WPOptim(
                l_net_locals[i].parameters(),
                base_optimizer=optim.SGD,
                lr=args.base_lr,
                momentum=0.9,
                weight_decay=5e-4,
            )
        if args.resume:
            optimizer.load_state_dict(checkpoint["sup_optimizers"][i])
        l_optim_locals.append(copy.deepcopy(optimizer.state_dict()))

    # 初始化无监督学习客户端
    u_weight_locals = []
    u_trainer_locals = []
    u_net_locals = []
    u_optim_locals = []

    for i in unsupervised_user_id:
        u_trainer_locals.append(
            UnsupervisedLocalUpdate(args, n_classes)
        )
        # w_locals.append(copy.deepcopy(w_glob))
        u_weight_locals.append(copy.deepcopy(weight_glob))
        u_net_locals.append(copy.deepcopy(net_glob))
        if args.opt == "adam":
            optimizer = torch.optim.Adam(
                u_net_locals[i - sup_num].parameters(),
                lr=args.unsup_lr,
                betas=(0.9, 0.999),
                weight_decay=5e-4,
            )
        elif args.opt == "sgd":
            optimizer = torch.optim.SGD(
                u_net_locals[i - sup_num].parameters(),
                lr=args.unsup_lr,
                momentum=0.9,
                weight_decay=5e-4,
            )
        elif args.opt == "adamw":
            optimizer = torch.optim.AdamW(
                u_net_locals[i - sup_num].parameters(),
                lr=args.unsup_lr,
                weight_decay=0.02,
            )
        elif args.opt == "wpoptim":
            # optimizer = torch.optim.SGD(u_net_locals[i - sup_num].parameters(),
            #                             lr=args.unsup_lr, momentum=0.9,
            #                             weight_decay=5e-4)
            optimizer = WPOptim(
                u_net_locals[i - sup_num].parameters(),
                base_optimizer=optim.SGD,
                lr=args.unsup_lr,
                momentum=0.9,
                weight_decay=5e-4,
            )
        if args.resume and len(checkpoint["unsup_optimizers"]) != 0:
            optimizer.load_state_dict(checkpoint["unsup_optimizers"][i - sup_num])
        u_optim_locals.append(copy.deepcopy(optimizer.state_dict()))

        if (
            args.resume
            and len(checkpoint["unsup_ema_state_dict"]) != 0
            and not args.from_labeled
        ):
            w_ema_unsup = copy.deepcopy(checkpoint["unsup_ema_state_dict"])
            u_trainer_locals[i - sup_num].ema_model.load_state_dict(
                w_ema_unsup[i - sup_num]
            )
            u_trainer_locals[i - sup_num].flag = False
            print("Unsup EMA reloaded")

    # for com_round in trange(1):
    #     #print("************* Communication round %d begins *************" % com_round)
    #     #print('upper bound')
    #     #print(f'upper bound of CNNs')
    #     w_l = []
    #     n_l = []

    #     for client_idx in supervised_user_id:
    #         noise_level = args.noise
    #         if client_idx == args.n_parties - 1:
    #             noise_level = 0
    #         noise_level = args.noise / (args.n_parties - 1) * client_idx
    #         loss_locals = []
    #         clt_this_comm_round = []
    #         w_per_meta = []
    #         local = l_trainer_locals[client_idx]
    #         optimizer = l_optim_locals[client_idx]
    #         #optimizer_trans = sup_optim_locals_trans[client_idx]

    #         #X_new, y_new = select_samlple(X_train[net_dataidx_map[client_idx]],y_train[net_dataidx_map[client_idx]],10)
    #         train_dl_local, train_ds_local = get_dataloader(args, X_train[net_dataidx_map[client_idx]],
    #                                                         y_train[net_dataidx_map[client_idx]],
    #                                                         args.dataset, args.datadir, args.batch_size,
    #                                                         is_labeled=True,
    #                                                         data_idxs=net_dataidx_map[client_idx],
    #                                                         pre_sz=args.pre_sz, input_sz=args.input_sz, noise_level=noise_level)
    #         w, loss, op = local.train(args, l_net_locals[client_idx].state_dict(),
    #                                            optimizer,
    #                                   train_dl_local, n_classes,stage=1)  # network, loss, optimizer
    #         w_l.append(w)
    #         n_l.append(len(net_dataidx_map[client_idx]))
    #         l_optim_locals[client_idx] = copy.deepcopy(op)
    #     w = FedAvg(w_l, n_l)

    #     net_glob.load_state_dict(w)
    #     if com_round%10==0:
    #         AUROC_avg, Accus_avg = test(com_round, net_glob.state_dict(), X_test, y_test, n_classes)
    #         print(AUROC_avg, Accus_avg)
    #         # print('adding lambda')
    #     for i in supervised_user_id:
    #         l_net_locals[i].load_state_dict(w)

    # net_glob.load_state_dict(w)
    # torch.save({'state_dict': net_glob.state_dict()},
    #            'test.pth')
    # AUROC_avg, Accus_avg = test(com_round, net_glob.state_dict(), X_test, y_test, n_classes)
    # print(AUROC_avg, Accus_avg)
    # load supervised pretrained models
    # state = torch.load('test.pth')
    # w = state['state_dict']
    # record_w = copy.deepcopy(w)
    # net_glob.load_state_dict(w)
    # for i in supervised_user_id:
    #     l_net_locals[i].load_state_dict(w)
    # for i in unsupervised_user_id:
    #     u_net_locals[i - sup_num].load_state_dict(w)
    # reweight

    if args.dataset == "office_caltech":
        reweight_para = 4
    else:
        sup_data_num = sum([len(net_dataidx_map[i]) for i in supervised_user_id])
        unsup_data_num = sum([len(net_dataidx_map[i]) for i in unsupervised_user_id])
        total_data_num = sup_data_num + unsup_data_num
        reweight_para = (
            int(unsup_data_num / sup_data_num)
            if (unsup_data_num / sup_data_num) > 1
            else 1
        )

    progress_bar = tqdm(range(start_epoch, args.rounds), desc="Training")
    for com_round in progress_bar:
        # print("************* Comm round %d begins *************" % com_round)
        logger.info("************* Comm round %d begins *************" % com_round)

        local_num = []
        local_w = []
        # local training
        for client_idx in supervised_user_id:
            noise_level = args.noise
            if client_idx == args.n_parties - 1:
                noise_level = 0
            noise_level = args.noise / (args.n_parties - 1) * client_idx

            trainer = l_trainer_locals[client_idx]
            optimizer = l_optim_locals[client_idx]
            if args.dataset == "office_caltech":
                train_dl_local = train_loaders[client_idx]
                train_ds_local = train_sets[client_idx]
            else:
                train_dl_local, train_ds_local = get_dataloader(
                    args,
                    X_train[net_dataidx_map[client_idx]],
                    y_train[net_dataidx_map[client_idx]],
                    args.dataset,
                    args.datadir,
                    args.batch_size,
                    is_labeled=True,
                    data_idxs=net_dataidx_map[client_idx],
                    pre_sz=args.pre_sz,
                    input_sz=args.input_sz,
                    noise_level=noise_level,
                )
            w, loss, op = trainer.train(
                args,
                l_net_locals[client_idx].state_dict(),
                optimizer,
                train_dl_local,
                n_classes,
            )  # network, loss, optimizer
            writer.add_scalar(
                "Supervised loss on sup client %d" % client_idx,
                loss,
                global_step=com_round,
            )
            local_w.append(w)
            l_optim_locals[client_idx] = copy.deepcopy(op)
            if args.dataset == "office_caltech":
                local_num.append(len(train_ds_local) * reweight_para)
            else:
                local_num.append(len(net_dataidx_map[client_idx]) * reweight_para)
        for client_idx in unsupervised_user_id:
            noise_level = args.noise
            if client_idx == args.n_parties - 1:
                noise_level = 0
            noise_level = args.noise / (args.n_parties - 1) * client_idx
            trainer = u_trainer_locals[client_idx - sup_num]
            optimizer = u_optim_locals[client_idx - sup_num]
            if args.dataset == "office_caltech":
                train_dl_local = train_loaders[client_idx]
                train_ds_local = train_sets[client_idx]
            else:
                train_dl_local, train_ds_local = get_dataloader(
                    args,
                    X_train[net_dataidx_map[client_idx]],
                    y_train[net_dataidx_map[client_idx]],
                    args.dataset,
                    args.datadir,
                    args.batch_size,
                    is_labeled=False,
                    data_idxs=net_dataidx_map[client_idx],
                    pre_sz=args.pre_sz,
                    input_sz=args.input_sz,
                    noise_level=noise_level,
                )
            w, loss, op, ratio, correct_pseu, all_pseu, train_right, same_pred_num = (
                trainer.train(
                    args,
                    u_net_locals[client_idx - sup_num].state_dict(),
                    tea_glob.state_dict(),
                    optimizer,
                    com_round * args.local_unsup_ep,
                    client_idx,
                    train_dl_local,
                    n_classes,
                )
            )
            local_w.append(w)
            u_optim_locals[client_idx - sup_num] = copy.deepcopy(op)
            if args.dataset == "office_caltech":
                local_num.append(len(train_ds_local))
            else:
                local_num.append(len(net_dataidx_map[client_idx]))
            writer.add_scalar(
                "Unsupervised loss on unsup client %d" % client_idx,
                loss,
                global_step=com_round,
            )
            logger.info(
                "Unlabeled client {} sample num: {} Training loss: {}, unsupervised loss ratio: {}, lr {}, {} pseu out of {} are correct, {} by model during train, total {}".format(
                    client_idx,
                    len(train_ds_local),
                    loss,
                    ratio,
                    u_optim_locals[client_idx - sup_num]["param_groups"][0]["lr"],
                    correct_pseu,
                    all_pseu,
                    train_right,
                    len(train_ds_local),
                )
            )
        with torch.no_grad():
            # aggregation phase
            w_glob = FedAvg(local_w, local_num)
            # update global mode
            net_glob.load_state_dict(w_glob)
            # for module in net_glob.modules():
            #     if isinstance(module, AmpNorm):
            #         module.fix_amp = True
            w_server = net_glob.state_dict()
            # distribute global model to all local models
            for i in supervised_user_id:
                l_net_locals[i].load_state_dict(w_server)
                # for module in l_net_locals[i].modules():
                #     if isinstance(module, AmpNorm):
                #         module.fix_amp = True
            for i in unsupervised_user_id:
                u_net_locals[i - sup_num].load_state_dict(w_server)
                # for module in u_net_locals[i - sup_num].modules():
                #     if isinstance(module, AmpNorm):
                #         module.fix_amp = True
            # tea_gloab update
            update_ema_variables(net_glob, tea_glob, args.ema_decay, com_round)
            logger.info(
                "************* Communication round {} ends *************".format(
                    com_round
                )
            )
            if args.dataset == "office_caltech":
                AUROC_avg, Accus_avg, *_ = test(
                    com_round,
                    net_glob.state_dict(),
                    n_classes=n_classes,
                    test_loaders=test_loaders,
                )
            else:
                AUROC_avg, Accus_avg, *_ = test(
                    com_round, net_glob.state_dict(), X_test, y_test, n_classes
                )
            progress_bar.set_postfix({"AUC": AUROC_avg, "Acc": Accus_avg})
            writer.add_scalar("AUC", AUROC_avg, global_step=com_round)
            writer.add_scalar("Acc", Accus_avg, global_step=com_round)
            logger.info("TEST Student: Epoch: {}".format(com_round))
            logger.info(
                "TEST AUROC: {:6f}, TEST Accus: {:6f}".format(AUROC_avg, Accus_avg)
            )

    final_model_path = os.path.join(snapshot_path, f"final_model_{time_current}.pth")
    torch.save({"state_dict": net_glob.state_dict()}, final_model_path)
    logger.info(f"Final model saved at {final_model_path}")
