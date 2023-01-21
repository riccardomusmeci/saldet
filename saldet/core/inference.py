import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from saldet.utils import *
from saldet.io import load_config
from saldet.model import create_model
from torch.utils.data import DataLoader
from saldet.dataset import InferenceDataset 
from saldet.transform import SaliencyTransform
from saldet.lightning_module import SaliencyModel

def inference(args):
    
    config = load_config(path=args.config)
    
    if args.ckpt.endswith(".pth"):
        model = create_model(
            model_name=config["model"]["model_name"],
            checkpoint_path=args.ckpt
        )
    elif args.ckpt.endswith(".ckpt"):
        model = SaliencyModel.load_from_checkpoint(
            checkpoint_path=args.ckpt, 
            map_location=device()
        )
    else:
        print(f"> [ERROR] {os.path.splitext(args.ckpt)[-1]} not supported.")
        quit()
    
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"> Loading images dataset")
    dataset = InferenceDataset(
        root_dir=args.images_dir,
        transform=SaliencyTransform(
            train=False,
            **config["transform"]
        )
    )
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=16
    )
    
    print(f"> Running inference on images and saving saliency maps at {args.output_dir}")
    model.to(device())
    model.eval()
    with torch.no_grad():
        for _, batch in tqdm(enumerate(data_loader), total=len(data_loader), desc="Inference"):
            x, image_names = batch
            x = x.to(device())
            preds = model(x)
            if hasattr(preds, "__iter__"):
                preds = preds[0]
            
            for pred, image_name in zip(preds, image_names):
                file_path = os.path.join(args.output_dir, f"{image_name}.png")
                pred = pred.squeeze().cpu().numpy()
                if args.thresh is not None:
                    pred[pred>=args.thresh] = 255
                    pred[pred<args.thresh] = 0
                else:
                    pred *= 255
                pred = Image.fromarray(pred).convert("L")
                pred.save(file_path)
        
            
            
        
        
    
    
    