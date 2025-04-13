from torch.utils.data import Dataset
import copy
import torch
import torch.optim
import torch.nn.functional as F
from options import args_parser
from models import ModelFedCon
import math

# from utils import losses, ramps
import torch.nn as nn
from utils import (
    label_guessing,
    sharpen,
    update_ema_variables,
    LinearRampUp,
    softmax_mse_loss,
    WPOptim,
    info_nce_loss,
)

# from loss.loss import UnsupervisedLoss  # , build_pair_loss
import logging
from torchvision import transforms

# from ramp import LinearRampUp
import logging

args = args_parser()
# 设置logger
logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


# 定义 ClassAwareBalance 类
class ClassAwareBalance:
    def __init__(self, net_proxy, threshold=0.4, t_l=5, t_h=6):
        self.net_proxy = net_proxy
        self.threshold = threshold
        self.t_l = t_l
        self.t_h = t_h
        self.softmax = nn.Softmax(dim=1)

    def cal_reweight(self, pred_label, probs_str):
        """
        计算重权掩码。

        参数:
        - pred_label (torch.Tensor): 预测标签，形状为 (batch_size,)。
        - probs_str (torch.Tensor): 类别概率分布，形状为 (batch_size, num_classes)。

        返回:
        - combined_mask (torch.Tensor): 每个样本的权重，形状为 (2B,)。
        - num_selected (int): 选择的样本数量。
        - num_unselected (int): 未选择的样本数量。
        """
        self.net_proxy.eval()
        with torch.no_grad():
            probs_str_sum = torch.sum(probs_str, dim=0)

            min_val = torch.min(probs_str_sum)
            max_val = torch.max(probs_str_sum)
            logits_sum_normalized = (probs_str_sum - min_val) / (
                max_val - min_val + 1e-8
            )
            confident_class = logits_sum_normalized > self.threshold
            confident_class_indices = torch.where(confident_class)[0]
            # Perform argsort operation on probs_str
            sorted_probs_indices = torch.argsort(probs_str, dim=1, descending=True)

            selected_positions = []
            for i, indices in enumerate(sorted_probs_indices):
                max_rank = float("inf")
                for class_index in confident_class_indices:
                    class_positions = (indices == class_index).nonzero(as_tuple=True)[0]
                    if class_positions.numel() > 0:
                        class_position = class_positions[0].item()
                        if class_position < max_rank:
                            max_rank = class_position
                # Check if the highest rank of the confident classes is within the range [t_l, t_h]
                if self.t_l <= max_rank <= self.t_h:
                    selected_positions.append(i)
            selected_positions = torch.tensor(selected_positions, dtype=torch.long).to(
                probs_str.device
            )
            # sample_positions = torch.where(
            #     torch.isin(pred_label, confident_class_indices))[0]

            # sample_probs = probs_str[sample_positions]
            # num_classes_above_threshold = (sample_probs > 0.4).sum(dim=1)
            # selected_positions = sample_positions[(
            #     num_classes_above_threshold >= self.t_l) & (num_classes_above_threshold <= self.t_h)]
            mask = torch.ones(probs_str.shape[0], dtype=torch.bool).to(probs_str.device)
            mask[selected_positions] = False
            unselected_positions = torch.where(mask)[0]

            num_selected = selected_positions.numel()
            num_unselected = probs_str.size(0) - num_selected
            unrelia_ctrl = 0.0 if num_selected == 0 else math.exp(-num_selected / 16.0)
            relia_ctrl = (
                0.0 if num_unselected == 0 else math.exp(-num_unselected / 16.0)
            )
            mask_unrelia = self.create_mask(
                probs_str.size(0), selected_positions, unrelia_ctrl
            )
            mask_relia = self.create_mask(
                probs_str.size(0), unselected_positions, relia_ctrl
            )

            # 拼接两个视角的掩码
            mask_unrelia = torch.cat([mask_unrelia, mask_unrelia], dim=0)  # (2B,)
            mask_relia = torch.cat([mask_relia, mask_relia], dim=0)  # (2B,)

            # 结合两个掩码的权重
            combined_mask = mask_unrelia + mask_relia  # (2B,)
            # 确保权重不为负
            combined_mask = combined_mask.clamp(min=0)

            return combined_mask, num_selected, num_unselected

    def create_mask(self, size, positions, ctrl):
        mask = torch.zeros(size, dtype=torch.float32).to(positions.device)
        mask[positions] = ctrl
        return mask


class UnsupervisedLocalUpdate(object):
    def __init__(self, args, n_classes):
        net = ModelFedCon(
            args.input_shape, args.model, args.out_dim, n_classes=n_classes
        )
        net_ema_local = ModelFedCon(
            args.input_shape, args.model, args.out_dim, n_classes=n_classes
        )
        net_ema_global = ModelFedCon(
            args.input_shape, args.model, args.out_dim, n_classes=n_classes
        )
        # net_proxy = ModelFedCon(args.model, args.out_dim, n_classes=n_classes)

        if len(args.gpu.split(",")) > 1:
            net = torch.nn.DataParallel(
                net, device_ids=[i for i in range(round(len(args.gpu) / 2))]
            )
            net_ema_local = torch.nn.DataParallel(
                net_ema_local, device_ids=[i for i in range(round(len(args.gpu) / 2))]
            )
            net_ema_global = torch.nn.DataParallel(
                net_ema_global, device_ids=[i for i in range(round(len(args.gpu) / 2))]
            )
            # net_proxy = torch.nn.DataParallel(
            #     net_proxy, device_ids=[i for i in range(round(len(args.gpu) / 2))])
        self.ema_model_local = net_ema_local.cuda()
        self.ema_model_global = net_ema_global.cuda()
        self.model = net.cuda()

        for param in self.ema_model_local.parameters():
            param.detach_()
        #self.data_idxs = idxs
        self.epoch = 0
        self.iter_num = 0
        self.flag = True
        self.unsup_lr = args.unsup_lr
        self.softmax = nn.Softmax()
        self.max_grad_norm = args.max_grad_norm
        # self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        #self.max_step = args.rounds * round(len(self.data_idxs) / args.batch_size)
        if args.opt == "adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=args.unsup_lr,
                betas=(0.9, 0.999),
                weight_decay=5e-4,
            )
        elif args.opt == "sgd":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=args.unsup_lr,
                momentum=0.9,
                weight_decay=5e-4,
            )
        elif args.opt == "adamw":
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=args.unsup_lr, weight_decay=0.02
            )
        elif args.opt == "wpoptim":
            # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=args.unsup_lr, momentum=0.9,
            #                                  weight_decay=5e-4)
            self.optimizer = WPOptim(
                self.model.parameters(),
                base_optimizer=torch.optim.SGD,
                lr=args.unsup_lr,
                momentum=0.9,
                weight_decay=5e-4,
            )
        self.max_warmup_step = args.num_warmup_epochs
        self.ramp_up = LinearRampUp(
            length=self.max_warmup_step, alpha_0=0.01, alpha_t=0.1
        )

    def train(
        self,
        args,
        net_w,
        net_tea_glob,
        op_dict,
        epoch,
        unlabeled_idx,
        train_dl_local,
        n_classes,
    ):
        self.model.load_state_dict(copy.deepcopy(net_w))
        self.ema_model_global.load_state_dict(copy.deepcopy(net_tea_glob))
        self.model.train()
        self.ema_model_local.eval()
        self.ema_model_global.eval()

        self.model.cuda()
        self.ema_model_local.cuda()
        # self.ema_model_global.cuda()

        self.optimizer.load_state_dict(op_dict)

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.unsup_lr

        self.epoch = epoch
        ramp_up_value = self.ramp_up(current=self.epoch)
        if self.flag:
            self.ema_model_local.load_state_dict(copy.deepcopy(net_w))
            self.flag = False
            logging.info("Local EMA model initialized")

        epoch_loss = []
        logging.info("Unlabeled client %d begin unsupervised training" % unlabeled_idx)
        correct_pseu = 0
        all_pseu = 0
        # test_right = 0
        # test_right_ema = 0
        train_right = 0
        same_total = 0
        for epoch in range(args.local_unsup_ep):
            batch_loss = []
            relia_num_cal = []
            unrelia_num_cal = []

            for i, (_, weak_aug_batch, label_batch) in enumerate(train_dl_local):
                weak_aug_batch = [
                    weak_aug_batch[version].cuda()
                    for version in range(len(weak_aug_batch))
                ]
                with torch.no_grad():
                    feature_l, guessed_l = label_guessing(
                        self.ema_model_local, [weak_aug_batch[0]], args.model
                    )
                    sharpened_l = sharpen(guessed_l)
                with torch.no_grad():
                    feature_g, guessed_g = label_guessing(
                        self.ema_model_global, [weak_aug_batch[0]], args.model
                    )
                    sharpened_g = sharpen(guessed_g)
                # print(f'feautre_l{feature_l.size()}')

                pseu = torch.argmax(sharpened_l, dim=1)
                label = label_batch.squeeze()
                if len(label.shape) == 0:
                    label = label.unsqueeze(dim=0)

                correct_pseu += torch.sum(
                    label[
                        (
                            torch.max(sharpened_l, dim=1)[0] > args.confidence_threshold
                        ).cpu()
                    ]
                    == pseu[
                        torch.max(sharpened_l, dim=1)[0] > args.confidence_threshold
                    ].cpu()
                ).item()
                all_pseu += len(
                    pseu[torch.max(sharpened_l, dim=1)[0] > args.confidence_threshold]
                )
                train_right += sum(
                    [
                        pseu[i].cpu() == label_batch[i].int()
                        for i in range(label_batch.shape[0])
                    ]
                )

                _, feature_c, logits_str = self.model(
                    weak_aug_batch[1], model=args.model
                )
                # feature_c = self.model(
                #     weak_aug_batch[1], model=args.model)[1]
                probs_str = F.softmax(logits_str, dim=1)
                pred_label = torch.argmax(logits_str, dim=1)
                same_total += sum(
                    [pred_label[sam] == pseu[sam] for sam in range(logits_str.shape[0])]
                )
                # cba
                cba = ClassAwareBalance(self.model)
                mask, unrelia_num, relia_num = cba.cal_reweight(pred_label, probs_str)
                # mask_unrelia = mask_unrelia.cuda()
                # mask_relia = mask_relia.cuda()
                unrelia_num_cal.append(unrelia_num)
                relia_num_cal.append(relia_num)
                # print(f'prob_str{probs_str}, sharpened_l{sharpened_l}')
                loss_l = info_nce_loss(feature_c, feature_l, masks=mask)
                # print(f'los   s_u: {loss_l}')
                loss_g = info_nce_loss(feature_c, feature_g, masks=mask)
                log_feature_c = F.log_softmax(feature_c, dim=1)
                soft_tea_local = F.softmax(feature_l, dim=1)
                soft_tea_global = F.softmax(feature_g, dim=1)
                V_l = F.kl_div(log_feature_c, soft_tea_local, reduction="batchmean")
                V_g = F.kl_div(log_feature_c, soft_tea_global, reduction="batchmean")
                # print(f'V_l is {V_l}, V_g is {V_g}')
                loss = ramp_up_value * (
                    torch.exp(-V_l) * loss_l * args.lambda_1
                    + torch.exp(-V_g) * loss_g * args.lambda_2
                )
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.generate_delta(zero_grad=True)
                
                # 第二次传播
                _, feature_c, logits_str = self.model(
                    weak_aug_batch[1], model=args.model
                )
                probs_str = F.softmax(logits_str, dim=1)
                pred_label = torch.argmax(logits_str, dim=1)
                # cba
                cba = ClassAwareBalance(self.model)
                mask, unrelia_num, relia_num = cba.cal_reweight(pred_label, probs_str)
                # mask_unrelia = mask_unrelia.cuda()
                # mask_relia = mask_relia.cuda()
                unrelia_num_cal.append(unrelia_num)
                relia_num_cal.append(relia_num)
                # print(f'prob_str{probs_str}, sharpened_l{sharpened_l}')
                loss_l = info_nce_loss(feature_c, feature_l, masks=mask)
                # print(f'los   s_u: {loss_l}')
                loss_g = info_nce_loss(feature_c, feature_g, masks=mask)
                log_feature_c = F.log_softmax(feature_c, dim=1)
                soft_tea_local = F.softmax(feature_l, dim=1)
                soft_tea_global = F.softmax(feature_g, dim=1)
                V_l = F.kl_div(log_feature_c, soft_tea_local, reduction="batchmean")
                V_g = F.kl_div(log_feature_c, soft_tea_global, reduction="batchmean")
                # print(f'V_l is {V_l}, V_g is {V_g}')
                loss_2ed = ramp_up_value * (
                    torch.exp(-V_l) * loss_l * args.lambda_1
                    + torch.exp(-V_g) * loss_g * args.lambda_2
                )
                loss_2ed.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=self.max_grad_norm
                )
                self.optimizer.step()

                update_ema_variables(
                    self.model, self.ema_model_local, args.ema_decay, self.iter_num
                )

                batch_loss.append(loss_2ed.item())

                self.iter_num = self.iter_num + 1

            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            logger.info(
                f"unrelia_num: {sum(unrelia_num_cal)}, relia_num: {sum(relia_num_cal)}"
            )
            self.epoch = self.epoch + 1
        self.model.cpu()
        self.ema_model_local.cpu()
        return (
            self.model.state_dict(),
            sum(epoch_loss) / len(epoch_loss),
            copy.deepcopy(self.optimizer.state_dict()),
            ramp_up_value,
            correct_pseu,
            all_pseu,
            train_right.cpu().item(),
            same_total.cpu().item(),
        )
