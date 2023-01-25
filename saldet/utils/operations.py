import torch
import torch.nn.functional as F

def flat(mask: torch.Tensor) -> torch.tensor:
    batch_size = mask.shape[0]
    h = 28
    mask = F.interpolate(mask,size=(int(h),int(h)), mode='bilinear')
    x = mask.view(batch_size, 1, -1).permute(0, 2, 1) 
    # print(x.shape)  b 28*28 1
    g = x @ x.transpose(-2,-1) # b 28*28 28*28
    g = g.unsqueeze(1) # b 1 28*28 28*28
    return g