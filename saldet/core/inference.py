import os
import torch
from PIL import Image
from tqdm import tqdm
from saldet.utils import *
import torch.nn.functional as F
from saldet.io import load_config
from torch.utils.data import DataLoader
from saldet.model import load_checkpoint
from saldet.dataset import InferenceDataset 
from saldet.utils.operations import normalize
from saldet.transform import SaliencyTransform

def inference(args):
    
    config = load_config(path=args.config)
    
    model = load_checkpoint(
        ckpt=args.ckpt,
        model_name=config["model"]["model_name"]
    )

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
        batch_size=1
    )
    
    print(f"> Running inference on images and saving saliency maps at {args.output_dir}")
    model.to(device())
    model.eval()
    with torch.no_grad():
        for _, batch in tqdm(enumerate(data_loader), total=len(data_loader), desc="Inference"):
            x, shape, image_names = batch
            x = x.to(device())
            preds = model(x)            
            if hasattr(preds, "__iter__"):
                preds = preds[0]
            if args.normalize:
                preds = normalize(preds)
            if args.interpolate:
                preds = F.interpolate(preds, size=shape[:2], mode="bilinear")
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
        
            
            
        
        
    
    
    