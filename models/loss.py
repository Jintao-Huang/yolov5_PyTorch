# Author: Jintao Huang
# Date: 

import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import box_ciou, cxcywh2ltrb


# yolov5默认不使用，但这个损失还是需要了解一下(Yolov5 is not used by default)
def binary_focal_loss(
        pred: torch.Tensor,
        target: torch.Tensor,
        alpha: float = 0.25,
        gamma: float = 2,
        reduction: str = "none",
) -> torch.Tensor:
    """Original implementation from https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py .
    Paper: https://arxiv.org/abs/1708.02002.
    公式：FocalLoss = alpha * (1 - p_t) ^ gamma * ce_loss. CELoss = -log(pred) * target

    :param pred: shape = (N,). 未过sigmoid
    :param target: shape = (N,)
    :param alpha: float. Weighting factor in range (0,1) to balance. alpha = -1(< 0) (no weighting)
    :param gamma: float
    :param reduction: 'none' | 'mean' | 'sum'
    :return: shape = ()
    """
    ce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
    p = torch.sigmoid(pred)
    p_t = target * p + (1 - target) * (1 - p)
    loss = ((1 - p_t) ** gamma) * ce_loss

    if alpha >= 0:
        alpha_t = target * alpha + (1 - target) * (1 - alpha)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


class Loss(nn.Module):
    def __init__(self, model, hyp):
        """

        :param hyp: dict["cls_pw", "obj_pw", "anchor_t"]
        """
        super(Loss, self).__init__()
        self.hyp = hyp
        self.balance = [4.0, 1.0, 0.4]  # 平衡各个layers的损失
        head = model.head
        self.num_layers, self.num_anchors = head.num_layers, head.num_anchors
        self.anchors = head.anchors  # shape[NL, NA, 2]

    def forward(self, pred, targets):
        """

        :param pred: e.g. Tuple[Tensor[N, 3, 28, 84, 25],
                                Tensor[N, 3, 14, 42, 25],
                                Tensor[N, 3, 7, 21, 25]]. [*cxcywh, obj_conf, cls_conf]
        :param targets: [idx, cls, *xywhn]
        :return:
        """
        device = targets.device
        cls_pw, obj_pw = torch.tensor([self.hyp['cls_pw'], self.hyp['obj_pw']], device=device)  # pos_weight
        cls_loss, box_loss, obj_loss = torch.zeros((3,), device=device)
        cls_t, box_t, indices, anchor_t = self._build_targets(pred, targets)
        for i, p in enumerate(pred):  # 每个layer
            # e.g. p.shape: [N, 3, 28, 84, 25]
            img_i, anchor_i, j_t, i_t = indices[i]
            obj_t = torch.zeros(p.shape[:4], dtype=torch.float32, device=device)
            num_targets = img_i.shape[0]
            if num_targets:
                p_sub = p[img_i, anchor_i, j_t, i_t]  # pred的子集

                # Regression
                """这里的a进行了归一化，与head中不同
                bx = 2 * sigmoid(tx) - 0.5 + ax. -0.5 ~ 1.5. ax = 0
                by = 2 * sigmoid(ty) - 0.5 + ay. ay = 0
                bw = (2 * sigmoid(tw)) ** 2 * aw. 0 ~ 4. aw / stride
                bh = (2 * sigmoid(th)) ** 2 * ah. ah / stride
                """
                # y[..., 0:2], self.grid[i], self.anchor_grid[i]
                # [N, 3, 48, 80, 2], [1, 1, 48, 80, 2], [1, 3, 1, 1, 2]
                xy_p = 2 * p_sub[:, 0:2].sigmoid() - 0.5  # center_xy
                wh_p = (2 * p_sub[:, 2:4].sigmoid()) ** 2 * anchor_t[i]  # wh
                box_p = torch.cat([xy_p, wh_p], dim=1)
                # box_p: shape[NT, 4], box_t[i]: shape[NT, 4]
                iou = box_ciou(cxcywh2ltrb(box_p), cxcywh2ltrb(box_t[i]))  #
                box_loss += (1 - iou).mean()

                # Objectness
                obj_t[img_i, anchor_i, j_t, i_t] = iou.detach().clamp_min(0)

                # Classification
                cls_t_hot = torch.zeros_like(p_sub[:, 5:], device=device)  # one_hot. float32. shape[NT, NC]
                cls_t_hot[torch.arange(num_targets), cls_t[i]] = 1.
                cls_loss += F.binary_cross_entropy_with_logits(p_sub[:, 5:], cls_t_hot, pos_weight=cls_pw)

            obj_loss += self.balance[i] * F.binary_cross_entropy_with_logits(p[..., 4], obj_t, pos_weight=obj_pw)
        box_loss *= self.hyp['box_lw']
        obj_loss *= self.hyp['obj_lw']
        cls_loss *= self.hyp['cls_lw']
        loss = box_loss + obj_loss + cls_loss
        batch_size = pred[0].shape[0]
        return loss * batch_size, torch.stack([box_loss, obj_loss, cls_loss, loss]).detach()

    def _build_targets(self, pred, targets):
        """选择正负样本

        :param pred: e.g. Tuple[Tensor[N, 3, 28, 84, 25],
                                Tensor[N, 3, 14, 42, 25],
                                Tensor[N, 3, 7, 21, 25]]. [*cxcywh, obj_conf, cls_conf]
        :param targets: [idx, cls, *xywhn]
        :return: cls_t, box_t, indices, anchor_t.
            cls_t: Len[3]. List[Tensor[NPos]]
            box_t: Len[3]. List[Tensor[NPos, 4]]
            indices: Len[3]. List[Tuple[
                img_i: Tensor[NPos],
                anchor_i: Tensor[NPos],
                j_t: Tensor[NPos], i_t: Tensor[NPos]]]
            anchor_t: Len[3]. List[Tensor[NPos, 2]]
        """
        device = targets.device
        # 每个layer的anchor数, layer数, target数
        num_layers, num_anchors, num_targets = self.num_layers, self.num_anchors, targets.shape[0]
        cls_t, box_t, indices, anchor_t = [], [], [], []
        # shape[NA, NT]
        anchor_i = torch.arange(num_anchors, dtype=torch.float32, device=device)[:, None].repeat(1, num_targets)
        # shape[NA, NT, 7]
        targets = torch.cat((targets[None].repeat(num_anchors, 1, 1), anchor_i[:, :, None]), 2)
        offset = torch.tensor([[0, 0],
                               [1, 0], [0, 1], [-1, 0], [0, -1],  # j, k, l, m
                               # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk, jm, lk, lm
                               ], dtype=torch.float32, device=device) * 0.5
        # 将targets从归一化空间变成grid空间
        gain = torch.ones(7, device=device)  # shape[targets.shape[-1]]
        for layer_i in range(num_layers):
            # self.anchors: shape[NL, NA, 2]
            anchors = self.anchors[layer_i]  # [NA, 2]
            # pred[0].shape. e.g. Tensor[N, 3, 28, 84, 25]
            gain[2:6] = torch.tensor(pred[layer_i].shape)[[3, 2, 3, 2]]  # whwh gain

            # Match targets to anchors
            # t: shape[NA, NT, 7]. [img_i, cls, *xywhn, anchor_i]
            t = targets * gain
            # targets 要去和 3个layers都去匹配
            if num_targets:
                # Matches
                ratio = t[:, :, 4:6] / anchors[:, None, :]  # w, h, shape[NA, NT, 2]
                # 第一个max: t太小太大都不行. 第二个max: 宽高比例都要符合
                pos_i = torch.max(ratio, 1. / ratio).max(dim=2)[0] < self.hyp['anchor_t']  # 正样本, shape[NA, NT]
                t = t[pos_i]  # shape[num_pos, 7]

                # Matches: 加入targets落入anchors感受域(-0.5 ~ 1.5)的anchor
                xy_t = t[:, 2:4]  # shape[num_pos, 2]
                xy_t_i = gain[2:4] - xy_t  # inverse. shape[num_pos, 2]
                # T的作用: shape[num_pos, 2] -> shape[2, num_pos]
                j, k = ((xy_t % 1. < 0.5) & (xy_t > 1.)).T  # 若xy_t < 0.5时, 不会吸引-1的anchor(不存在)
                l, m = ((xy_t_i % 1. < 0.5) & (xy_t_i > 1.)).T
                # pos_anchor_i[0]为落入的anchor(yolov3/v4)
                pos_i = torch.stack([torch.ones_like(j), j, k, l, m])  # 正样本. shape[5, NT]
                # [num_pos, 7] -> [5, num_pos, 7] -> [num_pos2, 7]
                t = t[None].repeat(5, 1, 1)[pos_i]
                # [5, 2] -> [5, num_pos, 2] -> [num_pos2, 2]
                off = offset[:, None, :].repeat(1, xy_t.shape[0], 1)[pos_i]
            else:
                t = t[0]  # shape[0, 7]
                off = 0
            img_i, cls = t[:, :2].long().T
            xy_t = t[:, 2:4]
            wh_t = t[:, 4:6]
            ij_t = (xy_t - off).long()
            i_t, j_t = ij_t.T
            anchor_i = t[:, 6].long()
            # Append
            cls_t.append(cls)  # class
            box_t.append(torch.cat([xy_t - ij_t, wh_t], 1))  # box
            indices.append((img_i, anchor_i,
                            j_t.clamp_(0, gain[3] - 1), i_t.clamp_(0, gain[2] - 1)))  # yx
            anchor_t.append(anchors[anchor_i])  # anchors

        return cls_t, box_t, indices, anchor_t
