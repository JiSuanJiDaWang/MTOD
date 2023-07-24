import torch.nn as nn
import torch
import torch.nn.functional as F

# Define l1 loss and l2 loss for depth estimation


class MaskedMSELoss(nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def forward(self, pred, target):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target > 0).detach()
        diff = target - pred
        diff = diff[valid_mask]
        self.loss = (diff ** 2).mean()
        return self.loss


class MaskedL1Loss(nn.Module):
    def __init__(self):
        super(MaskedL1Loss, self).__init__()

    def forward(self, pred, target):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target > 0).detach()
        diff = target - pred
        diff = diff[valid_mask]
        self.loss = diff.abs().mean()
        return self.loss


class MultiBoxLoss(nn.Module):
    def __init__(self, num_classes, alpha=1.0, neg_pos_ratio=3.0,
                 background_label_id=0, negative_for_hard=100.0):

        super(MultiBoxLoss, self).__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.neg_pos_ratio = neg_pos_ratio
        self.background_label_id = background_label_id
        self.negative_for_hard = torch.FloatTensor([negative_for_hard])[0]

    def _l1_smooth_loss(self, y_true, y_pred):
        abs_loss = torch.abs(y_true - y_pred)
        sq_loss = 0.5 * (y_true - y_pred)**2
        l1_loss = torch.where(abs_loss < 1.0, sq_loss, abs_loss - 0.5)
        return torch.sum(l1_loss, -1)

    def _softmax_loss(self, y_true, y_pred):
        y_pred = torch.clamp(y_pred, min=1e-7)
        softmax_loss = -torch.sum(y_true * torch.log(y_pred),
                                  axis=-1)
        return softmax_loss

    def forward(self, y_pred, y_gt):

        num_boxes = y_gt.size()[1]
        y_pred = torch.cat([y_pred[0], nn.Softmax(-1)(y_pred[1])], dim=-1)

        conf_loss = self._softmax_loss(y_gt[:, :, 4:-1], y_pred[:, :, 4:])

        loc_loss = self._l1_smooth_loss(y_gt[:, :, :4],
                                        y_pred[:, :, :4])

        pos_loc_loss = torch.sum(loc_loss * y_gt[:, :, -1],
                                 axis=1)
        pos_conf_loss = torch.sum(conf_loss * y_gt[:, :, -1],
                                  axis=1)

        num_pos = torch.sum(y_gt[:, :, -1], axis=-1)

        num_neg = torch.min(self.neg_pos_ratio * num_pos, num_boxes - num_pos)
        # 找到了哪些值是大于0的
        pos_num_neg_mask = num_neg > 0

        has_min = torch.sum(pos_num_neg_mask)

        num_neg_batch = torch.sum(
            num_neg) if has_min > 0 else self.negative_for_hard

        confs_start = 4 + self.background_label_id + 1
        confs_end = confs_start + self.num_classes - 1

        max_confs = torch.sum(y_pred[:, :, confs_start:confs_end], dim=2)

        max_confs = (max_confs * (1 - y_gt[:, :, -1])).view([-1])

        _, indices = torch.topk(max_confs, k=int(
            num_neg_batch.cpu().numpy().tolist()))

        neg_conf_loss = torch.gather(conf_loss.view([-1]), 0, indices)

        # 进行归一化
        num_pos = torch.where(num_pos != 0, num_pos, torch.ones_like(num_pos))
        total_loss = torch.sum(
            pos_conf_loss) + torch.sum(neg_conf_loss) + torch.sum(self.alpha * pos_loc_loss)
        total_loss = total_loss / torch.sum(num_pos)

        return total_loss


class MTloss(nn.Module):

    def __init__(self, num_classes):
        super(MTloss, self).__init__()
        # self.depthLoss1 = MaskedMSELoss()
        self.depthLoss = MaskedL1Loss()
        self.ssdLoss = MultiBoxLoss(num_classes)

    def forward(self, x_pred, label, task):

        # binary mark to mask out undefined pixel space
        if task == "detection":
            loss = self.ssdLoss(x_pred, label)

        if task == "depth":
            # device = x_pred.device
            # binary_mask = (torch.sum(label, dim=1) != 0).float().unsqueeze(1).to(device)

            # loss = self.depthLoss(x_pred, label)
            # loss = torch.sum(torch.abs(x_pred - label) * binary_mask) / torch.nonzero(binary_mask, as_tuple=False).size(0)

            # L1 Loss with Ignored Region (values are 0 or -1)
            invalid_idx = -1
            valid_mask = (torch.sum(label, dim=1, keepdim=True)
                          != invalid_idx).to(x_pred.device)
            loss = torch.sum(F.l1_loss(x_pred, label, reduction='none').masked_select(valid_mask)) \
                / torch.nonzero(valid_mask, as_tuple=False).size(0)

        return loss
