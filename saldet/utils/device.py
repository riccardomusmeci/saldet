import torch

def device():    
    if torch.cuda.is_available(): return "cuda"
    if torch.has_mps: return "mps"
    return "cpu"