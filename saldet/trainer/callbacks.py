import os
from typing import List

import pytorch_lightning
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint


def create_callbacks(
    output_dir: str,
    filename: str = "epoch={epoch}-step={step}-val_loss={loss/val:.3f}",
    monitor: str = "loss/val",
    mode: str = "min",
    save_top_k: int = 5,
    patience: int = 10,
) -> List[pytorch_lightning.Callback]:
    """Return list of callbacks for Trainer

    Args:
        output_dir (str): output dir
        filename (str, optional): checkpoint filename. Defaults to epoch={epoch}-step={step}-val_loss={loss/val:.3f}-val_iou={IoU/all/val:.3f}.
        monitor (str, optional): metric to monitor. Defaults to "loss/val".
        mode (str, optional): monitor mode. Defaults to "min".
        save_top_k (int, optional): number of best models to save based on monitor. Defaults to 5.
        patience (int, optional): early stopping patience. Defaults to 10.

    Returns:
        List[pytorch_lightning.Callback]: list of Callbacks.
    """
    callbacks = []
    callbacks.append(
        ModelCheckpoint(
            dirpath=os.path.join(output_dir, "checkpoints"),
            filename=filename,
            monitor=monitor,
            verbose=True,
            mode=mode,
            save_top_k=save_top_k,
        )
    )

    callbacks.append(
        EarlyStopping(
            monitor=monitor,
            min_delta=0.0,
            patience=patience,
            verbose=True,
            mode="min",
            check_finite=True,
            stopping_threshold=None,
            divergence_threshold=None,
            check_on_train_epoch_end=None,
        )
    )

    return callbacks
