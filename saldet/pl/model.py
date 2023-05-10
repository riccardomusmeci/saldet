from typing import Any, List, Tuple, Union

import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from saldet.utils import device


class SaliencyPLModel(pl.LightningModule):
    """Saliency Model with PyTorchLightning Module

    Args:
        model (nn.Module): saliency model
        criterion (_Loss): criterion for loss
        optimizer (Optimizer): optimizer
        lr_scheduler (_LRScheduler, optional): Optional; Learning rate scheduler. Defaults to None.
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: _Loss,
        optimizer: Optimizer,
        lr_scheduler: _LRScheduler = None,
    ) -> None:

        super().__init__()
        self.save_hyperparameters()
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        # TODO: add BinaryMetrics

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Output a tensor of shape (batch size, num classes, height, width)
        return self.model(x)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> float:
        """Training step

        Args:
            batch (torch.Tensor): batch of images and masks
            batch_idx (int): batch index

        Returns:
            float: loss value
        """
        x, mask = batch
        preds = self(x)
        # Loss
        loss = self.criterion(preds, mask)
        self.log("loss_train", loss, sync_dist=True)
        self.log("lr", self.lr_scheduler.get_last_lr()[0], prog_bar=True)
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        """Validation step

        Args:
            batch (torch.Tensor): batch of images and masks
            batch_idx (int): batch index

        """
        x, mask = batch
        preds = self(x)
        loss = self.criterion(preds, mask)

        self.log("loss_val", loss, sync_dist=True, prog_bar=True)

    def configure_optimizers(
        self,
    ) -> Union[List[Optimizer], Tuple[List[Optimizer], List[_LRScheduler]]]:
        if self.lr_scheduler is None:
            return [self.optimizer]
        else:
            return [self.optimizer], [self.lr_scheduler]
