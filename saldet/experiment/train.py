import os
from pathlib import Path
from shutil import copy

import pytorch_lightning as pl

from saldet import create_model
from saldet.io import load_config
from saldet.loss import create_criterion
from saldet.lr_scheduler import create_scheduler
from saldet.optimizer import create_optimizer
from saldet.pl import SaliencyPLDataModule, SaliencyPLModel
from saldet.trainer import create_callbacks
from saldet.transform import SaliencyTransform
from saldet.utils import now


def train(
    data_dir: Path,
    config_path: Path,
    output_dir: Path,
    resume_from: Path = None,
    seed: int = 42,
):
    """Train experiment entry point

    Args:
        data_dir (Path): path to data directory.
        config_path (Path): path to yaml configuration file
        output_dir (Path): path to output dir
        resume_from (Path, optional): ckpt from pytorch_lightning to resume training from. Defaults to None.
        seed (int, optional): reproducibility seed. Defaults to 42.
    """
    pl.seed_everything(seed=seed, workers=True)
    config = load_config(config_path)
    output_dir = os.path.join(output_dir, now())

    print(f"> Output dir will be {output_dir}")
    os.makedirs(output_dir)
    copy(config_path, os.path.join(output_dir, "config.yaml"))

    # datamodule
    datamodule = SaliencyPLDataModule(
        root_dir=data_dir,
        train_transform=SaliencyTransform(train=True, **config["transform"]),
        val_transform=SaliencyTransform(train=False, **config["transform"]),
        **config["datamodule"],
    )

    model = create_model(**config["model"])
    criterion = create_criterion(**config["loss"])
    optimizer = create_optimizer(params=model.parameters(), **config["optimizer"])
    lr_scheduler = create_scheduler(optimizer=optimizer, **config["lr_scheduler"])

    pl_model = SaliencyPLModel(
        model=model, criterion=criterion, optimizer=optimizer, lr_scheduler=lr_scheduler
    )

    callbacks = create_callbacks(output_dir=output_dir, **config["callbacks"])

    if resume_from is not None:
        print(f"> Resuming training from {resume_from}.")

    trainer = pl.Trainer(logger=False, callbacks=callbacks, **config["trainer"])

    # fit
    print(f"Launching training...")
    trainer.fit(model=pl_model, datamodule=datamodule, ckpt_path=resume_from)
