import os
import argparse
from shutil import copy
from saldet.utils *
from saldet.io import load_config
from saldet.model import create_model
from saldet.loss import create_criterion
from saldet.optimizer import create_optimizer
from saldet.transform import SaliencyTransform
from saldet.lr_scheduler import create_scheduler





def parse_args() -> argparse.Namespace:
    
    parser = argparse.ArgumentParser("Training config")
    
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        type=str,
        required=False,
        help="path to the YAML configuration file"
    )
    
    parser.add_argument(
        "--output-dir",
        default="/Users/riccardomusmeci/Developer/experiments/github/saliency-detection/DUTS/",
        type=str,
        help="local directory where the best model checkpoint is saved at the end of training."
    )
    
    parser.add_argument(
        "--data-dir",
        metavar="N",
        default="/Users/riccardomusmeci/Developer/data/github/saliency/DUTS",
        help="Input data dir path."
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    return args

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
    saliency_model = SaliencyModule(
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
    
    # trainer 
    trainer = pl.Trainer(
        logger=False,
        callbacks=callbacks,
        **config["trainer"]
    )
    
    # fit 
    print(f"Launching training...")
    trainer.fit(model=saliency_model, datamodule=datamodule)
    
if __name__ == "__main__":
    args = parse_args()
    train(args)