# Author: Jintao Huang
# Date: 2021-4-27

import torch.nn as nn
import torch

__all__ = ["Head"]


class Head(nn.Module):
    def __init__(self, num_classes, anchors=None):
        """

        :param num_classes: int
        :param anchors: Tuple[Tuple[W, H]].
        """
        super(Head, self).__init__()
        self.out_channels = num_classes + 5
        anchors = anchors or ((10, 13, 16, 30, 33, 23),
                              (30, 61, 62, 45, 59, 119),
                              (116, 90, 156, 198, 373, 326))
        # e.g. [1, 3, 48, 80, 2] * [1, 3, 1, 1, 2]
        self.num_layers, self.num_anchors = len(anchors), len(anchors[0]) // 2
        anchors = torch.tensor(anchors, dtype=torch.float32) \
            .view(self.num_layers, self.num_anchors, 2)  # shape[NL, NA, 2]
        # shape[NL, N, NA, H, W, 2]. 即 [NL, 1, NA, 1, 1, 2]
        anchors_grid = anchors.clone().view(self.num_layers, 1, self.num_anchors, 1, 1, 2)
        self.register_buffer("anchors", anchors)  # for loss.build_targets()
        self.register_buffer("anchors_grid", anchors_grid)  # for self.forward()
        self.grid = [torch.zeros(())] * self.num_layers
        self.stride = [8., 16., 32.]
        self.anchors = self.anchors / torch.tensor(self.stride, device=self.anchors.device)[:, None, None]
        self.conv_list = nn.ModuleList([
            nn.Conv2d(128, self.num_anchors * self.out_channels, 1, 1, 0, bias=True),
            nn.Conv2d(256, self.num_anchors * self.out_channels, 1, 1, 0, bias=True),
            nn.Conv2d(512, self.num_anchors * self.out_channels, 1, 1, 0, bias=True)
        ])

    def forward(self, x):
        """

        :param x: List[Tensor]. e.g. List[shape[N, 128, 80, 80], shape[N, 256, 40, 40], shape[N, 512, 20, 20]]
        :return: x: e.g. List[shape[N, 3, 80, 80, 25], shape[N, 3, 40, 40, 25], shape[N, 3, 20, 20, 25]]
                 z: e.g. shape[N, 15120, 25]. [cxcywh, obj_conf, cls_conf].
        """
        z = []  # detect output
        x = list(x)  # not tuple
        for i in range(len(x)):
            x[i] = self.conv_list[i](x[i])
            # e.g. [N, 75, 20, 20] -> [N, 3, 25, 20, 20] -> [N, 3, 20, 20, 25]
            x[i] = x[i].view(x[i].shape[0], 3, self.out_channels, *x[i].shape[-2:])
            x[i] = x[i].permute((0, 1, 3, 4, 2)).contiguous()
            if not self.training:  # test or detect
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(*x[i].shape[2:4]).to(x[i].device)
                """
                bx = 2 * sigmoid(tx) - 0.5 + ax. -0.5 ~ 1.5
                by = 2 * sigmoid(ty) - 0.5 + ay
                bw = (2 * sigmoid(tw)) ** 2 * aw. 0 ~ 4
                bh = (2 * sigmoid(th)) ** 2 * ah
                """
                y = x[i].sigmoid()  # y.shape: [N, 3, H, W, 25]
                # y[..., 0:2], self.grid[i], self.anchor_grid[i]
                # [N, 3, 48, 80, 2], [1, 1, 48, 80, 2], [1, 3, 1, 1, 2]
                y[..., 0:2] = (2 * y[..., 0:2] - 0.5 + self.grid[i]) * self.stride[i]  # center_xy
                y[..., 2:4] = (2 * y[..., 2:4]) ** 2 * self.anchors_grid[i]  # wh
                y = y.view(x[i].shape[0], -1, self.out_channels)
                z.append(y)

        return x if self.training else (torch.cat(z, dim=1), x)

    @staticmethod
    def _make_grid(ny, nx):
        """

        :param ny: int. nh. e.g. 48
        :param nx: int. nw. e.g. 80
        :return:
        """
        yv, xv = torch.meshgrid([torch.arange(ny, dtype=torch.float32),
                                 torch.arange(nx, dtype=torch.float32)])
        # e.g. [1, 3, 48, 80, 2] + [1, 1, 48, 80, 2]
        return torch.stack((xv, yv), 2).view(1, 1, ny, nx, 2)


if __name__ == "__main__":
    # test
    head = Head(25).cuda()
    x = torch.randn(2, 128, 48, 80).cuda(), torch.randn(2, 256, 24, 40).cuda(), torch.randn(2, 512, 12, 20).cuda()
    head.eval()
    head(x)
    x = head(x)
    print(1)
