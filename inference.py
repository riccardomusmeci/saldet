import os
import argparse
from saldet.core import inference

import warnings
warnings.filterwarnings("ignore", category=UserWarning) 

def parse_args() -> argparse.Namespace:
    
    parser = argparse.ArgumentParser("Inference config")
    
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="path to the YAML configuration file"
    )
    
    parser.add_argument(
        "--ckpt",
        required=True,
        type=str,
        help="local directory where the best model checkpoint is saved at the end of training"
    )
    
    parser.add_argument(
        "--images-dir",
        metavar="N",
        required=True,
        help="input images dir path."
    )
    
    parser.add_argument(
        "--output-dir",
        metavar="N",
        required=True,
        help="output data dir to save saliency maps"
    )
    
    parser.add_argument(
        "--thresh",
        default=None,
        type=float,
        required=False,
        help="saliency map threshold"
    )
    
    parser.add_argument(
        "--interpolate",
        default=True,
        type=lambda x: True if x.lower()=="true" else False,
        help="if mask from model is to be interpolated before saving it"
    )
    
    parser.add_argument(
        "--normalize",
        default=True,
        type=lambda x: True if x.lower()=="true" else False,
        help="if pred mask has to be normalized"
    )
    
    args = parser.parse_args()
    
    return args

if __name__ == "__main__":
    args = parse_args()
    inference(args)