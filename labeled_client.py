import numpy as np
import torch
import torch.optim
from options import args_parser
import copy

# from utils import losses
import logging

# from pytorch_metric_learning import losses
from models import ModelFedCon
from utils import WPOptim

args = args_parser()


class SupervisedLocalUpdate(object):
    def __init__(self, args, n_classes):
        self.epoch = 0
        self.iter_num = 0
        # self.confuse_matrix = torch.zeros((5, 5)).cuda()
        self.base_lr = args.base_lr
        #self.data_idx = idxs
        self.max_grad_norm = args.max_grad_norm
        temperature = 0.1  # tao in paper
        # self.cont_loss_func = losses.NTXentLoss(temperature)

        net = ModelFedCon(
            args.input_shape, args.model, args.out_dim, n_classes=n_classes
        )
        if len(args.gpu.split(",")) > 1:
            net = torch.nn.DataParallel(
                net, device_ids=[i for i in range(round(len(args.gpu) / 2))]
            )
        self.model = net.cuda()

    def train(self, args, net_w, op_dict, dataloader, n_classes, stage=2):
        self.model.load_state_dict(copy.deepcopy(net_w))
        self.model.cuda().train()
        if args.opt == "adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=args.base_lr,
                betas=(0.9, 0.999),
                weight_decay=5e-4,
            )
        elif args.opt == "sgd":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=args.base_lr,
                momentum=0.9,
                weight_decay=5e-4,
            )
        elif args.opt == "adamw":
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=args.base_lr, weight_decay=0.02
            )
        elif args.opt == "wpoptim":
            # self.optimizer = torch.optim.SGD(self.model.parameters(),
            #                                  lr=args.base_lr, momentum=0.9,
            #                                  weight_decay=5e-4)
            self.optimizer = WPOptim(
                self.model.parameters(),
                base_optimizer=torch.optim.SGD,
                lr=args.base_lr,
                momentum=0.9,
                weight_decay=5e-4,
            )
        # SimPLE original paper: lr=0.002, weight_decay=0.02
        self.optimizer.load_state_dict(op_dict)

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.base_lr

        loss_fn = torch.nn.CrossEntropyLoss()
        epoch_loss = []
        logging.info("Begin supervised training")
        if stage == 1:
            if args.dataset == "cifar100":
                args.local_sup_ep = 1001
            else:
                args.local_sup_ep = 10
        else:
            args.local_sup_ep = 9
        for epoch in range(args.local_sup_ep):
            batch_loss = []
            for i, (_, image_batch, label_batch) in enumerate(dataloader):

                image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
                label_batch = label_batch.long().squeeze()
                inputs = image_batch
                _, activations, outputs = self.model(inputs, model=args.model)

                if len(label_batch.shape) == 0:
                    label_batch = label_batch.unsqueeze(dim=0)
                if len(outputs.shape) != 2:
                    outputs = outputs.unsqueeze(dim=0)

                loss_classification = loss_fn(outputs, label_batch)
                loss = loss_classification
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.generate_delta(zero_grad=True)
                # 第二次传播
                _, activations, outputs = self.model(inputs, model=args.model)
                if len(label_batch.shape) == 0:
                    label_batch = label_batch.unsqueeze(dim=0)
                if len(outputs.shape) != 2:
                    outputs = outputs.unsqueeze(dim=0)

                loss_2nd = loss_fn(outputs, label_batch)
                loss_2nd.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=self.max_grad_norm
                )
                self.optimizer.step(zero_grad=True)
                batch_loss.append(loss_2nd.item())
                self.iter_num = self.iter_num + 1

            self.epoch = self.epoch + 1
            epoch_loss.append(np.array(batch_loss).mean())

        self.model.cpu()
        return (
            self.model.state_dict(),
            sum(epoch_loss) / len(epoch_loss),
            copy.deepcopy(self.optimizer.state_dict()),
        )
