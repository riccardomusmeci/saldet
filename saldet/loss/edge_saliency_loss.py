import torch
import torch.nn as nn
import torch.nn.functional as F

from saldet.utils.device import device


class EdgeSaliencyLoss(nn.Module):
    def __init__(self, alpha_sal=0.7):
        super(EdgeSaliencyLoss, self).__init__()

        self.alpha_sal = alpha_sal

        self.laplacian_kernel = torch.tensor(
            [[-1.0, -1.0, -1.0], [-1.0, 8.0, -1.0], [-1.0, -1.0, -1.0]],
            dtype=torch.float,
            requires_grad=False,
        )
        self.laplacian_kernel = self.laplacian_kernel.view(
            (1, 1, 3, 3)
        )  # Shape format of weight for convolution
        self.laplacian_kernel = self.laplacian_kernel.to(device())

    @staticmethod
    def weighted_bce(input_, target, weight_0=1.0, weight_1=1.0, eps=1e-15):
        wbce_loss = -weight_1 * target * torch.log(input_ + eps) - weight_0 * (
            1 - target
        ) * torch.log(1 - input_ + eps)
        return torch.mean(wbce_loss)

    def forward(self, y_pred, y_gt):
        # Generate edge mapsx
        y_gt_edges = F.relu(
            torch.tanh(F.conv2d(y_gt.float(), self.laplacian_kernel, padding=(1, 1)))
        )
        y_pred_edges = F.relu(
            torch.tanh(F.conv2d(y_pred, self.laplacian_kernel, padding=(1, 1)))
        )

        # sal_loss = F.binary_cross_entropy(input=y_pred, target=y_gt)
        sal_loss = self.weighted_bce(
            input_=y_pred, target=y_gt, weight_0=1.0, weight_1=1.12
        )
        edge_loss = F.binary_cross_entropy(input=y_pred_edges, target=y_gt_edges)

        total_loss = self.alpha_sal * sal_loss + (1 - self.alpha_sal) * edge_loss
        return total_loss
