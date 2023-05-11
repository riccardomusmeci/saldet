import os
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from saldet import ops
from saldet.dataset import InferenceDataset
from saldet.io import load_config
from saldet.model import load_checkpoint
from saldet.transform import SaliencyTransform
from saldet.utils import device


def inference(
    images_dir: Path,
    config_path: Path,
    ckpt: Path,
    output_dir: Path,
    normalize: bool = True,
    resize_to_original: bool = True,
    threshold: float = 0.5,
):
    """Inference entry point - create saliency maps and save them all in an output dir

    Args:
        images_dir (Path): path to folder with images
        config_path (Path): path to saldet configuration file
        ckpt (Path): path to model ckpt (.pth or .ckpt from pytorch_lightning)
        output_dir (Path): path to output dir where saliency maps will be saved
        normalize (bool, optional): if True, normalizes saliency maps. Defaults to True.
        resize_to_original (bool, optional): if True, resize_to_original saliency maps. Defaults to True.
        threshold (float, optional): if True, applies a threshold to saliency maps to be binary saliency maps. Defaults to .5.
    """
    config = load_config(path=config_path)

    model = load_checkpoint(ckpt=ckpt, model_name=config["model"]["model_name"])

    os.makedirs(output_dir, exist_ok=True)
    print(f"> Loading images dataset")
    dataset = InferenceDataset(
        root_dir=images_dir,
        transform=SaliencyTransform(train=False, **config["transform"]),
    )
    data_loader = DataLoader(dataset=dataset, batch_size=1)

    print(f"> Running inference on images and saving saliency maps at {output_dir}")
    model.to(device())
    model.eval()
    with torch.no_grad():
        for _, batch in tqdm(
            enumerate(data_loader), total=len(data_loader), desc="Inference"
        ):
            x, image_path = batch
            x = x.to(device())
            preds = model(x)
            if hasattr(preds, "__iter__"):
                preds = preds[0]
            if normalize:
                preds = ops.normalize(preds)
            if resize_to_original:
                w, h = Image.open(image_path[0]).size
                preds = F.interpolate(preds, size=(h, w), mode="bilinear")
            for pred, image_path in zip(preds, image_path):
                image_name = os.path.basename(image_path).split(".")[0]
                file_path = os.path.join(output_dir, f"{image_name}.png")
                pred = pred.squeeze().cpu().numpy()
                if threshold is not None:
                    pred[pred >= threshold] = 255
                    pred[pred < threshold] = 0
                else:
                    pred *= 255
                pred = Image.fromarray(pred).convert("L")
                pred.save(file_path)
