import os
from shutil import copy
from saldet.utils import *
import pytorch_lightning as pl
from saldet.io import load_config
from saldet.model import create_model
from saldet.loss import create_criterion
from saldet.optimizer import create_optimizer
from saldet.transform import SaliencyTransform
from saldet.lr_scheduler import create_scheduler
from saldet.lightning_module import SaliencyDataModule, SaliencyModel

def train(args):
    
    pl.seed_everything(seed=args.seed, workers=True)
    config = load_config(path=args.config)
    output_dir = os.path.join(args.output_dir, now())
    
    # copying config
    os.makedirs(output_dir)
    copy(args.config, os.path.join(output_dir, "config.yml"))
    
    # datamodule
    datamodule = SaliencyDataModule(
        root_dir=args.data_dir,
        train_transform=SaliencyTransform(train=True, **config["transform"]),
        val_transform=SaliencyTransform(train=False, **config["transform"]),
        **config["datamodule"]
    )
    
    # model, loss, optimizer, lr_scheduler
    model = create_model(**config["model"])
    criterion = create_criterion(**config["loss"])
    optimizer = create_optimizer(
        params=model.parameters(),
        **config["optimizer"]
    )
    lr_scheduler = create_scheduler(
        optimizer=optimizer,
        **config["lr_scheduler"]
    )
    
    # Model LightningModule
    saliency_model = SaliencyModel(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler
    )
    
    # Lightning Callbacks
    callbacks = create_callbacks(
        output_dir=output_dir,
        **config["callbacks"]
    )
    
    if args.resume_from is not None:
        print(f"> Resuming training from {args.resume_from}.")
        
    # trainer 
    trainer = pl.Trainer(
        logger=False,
        callbacks=callbacks,
        resume_from_checkpoint=args.resume_from,
        **config["trainer"]
    )
    
    # fit 
    print(f"Launching training...")
    trainer.fit(model=saliency_model, datamodule=datamodule)
    