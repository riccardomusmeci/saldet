import os
import torch
import argparse
from tqdm import tqdm
from shutil import copy
from saldet.utils import *
from saldet.io import load_config
from saldet.model import create_model
from saldet.loss import create_criterion
from saldet.data import create_dataloader
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
    
    # seed_everything(seed=args.seed)
    config = load_config(path=args.config)
    output_dir = os.path.join(args.output_dir, now())
    
    # copying config
    os.makedirs(output_dir)
    copy(args.config, os.path.join(output_dir, "config.yml"))
    
    print(f"> Loading Train Dataset")
    train_dl = create_dataloader(
        root_dir=args.data_dir,
        train=True,
        transform=SaliencyTransform(train=True, **config["transform"]),
        **config["data_loader"]
    )
    
    print(f"> Loading Val Dataset")
    val_dl = create_dataloader(
        root_dir=args.data_dir,
        train=False,
        transform=SaliencyTransform(train=False, **config["transform"]),
        **config["data_loader"]
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
    
    model.to(device=device())
    criterion.to(device=device())
    
    epochs = 500
    check_train_every_n_iters = 10
    print(f"> Starting training...")
    for epoch in range(epochs):
        
        print(f"> Epoch [{epoch}/{epochs}]")
        # train
        model.train()
        epoch_loss = 0.0
        for batch_idx, batch in enumerate(train_dl):
            x, target = (el.to(device()) for el in batch)
            # zero the parameter gradients
            optimizer.zero_grad()
            preds = model(x)
            loss = criterion(preds[0], target)
            loss.backward()
            optimizer.step()
            # print statistics
            epoch_loss += loss.item()
            if (batch_idx+1) % check_train_every_n_iters == 0:    # print every 50 mini-batches
                print("\t> epoch [{}/{}] - iter [{}/{}] - loss/train={} - lr={:.4f}".format(
                    epoch,
                    epochs,
                    batch_idx+1,
                    len(train_dl),
                    epoch_loss / check_train_every_n_iters,
                    lr_scheduler.get_last_lr()[0]
                ))
                epoch_loss = 0.0
                
        # validation
        print(f"> Validation step...")
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for batch_idx, batch in tqdm(enumerate(val_dl), total=len(val_dl)):
                x, target = (el.to(device()) for el in batch)        
                preds = model(x)
                loss = criterion(preds[0], target)
                
                # print statistics
                val_loss += loss.item()
            val_loss = val_loss / len(val_dl)
            print("\t> loss/val={:.4f}".format(val_loss))
                    
        lr_scheduler.step()
        
        file_path = os.path.join(
            output_dir,
            f"model-val_loss={val_loss:.4f}.pth"
        )
        print(f"> Saving pth at  {file_path}")
        torch.save(
            model.state_dict(),
            file_path
        )
        
if __name__ == "__main__":
    args = parse_args()
    train(args)