from torch.utils.data import DataLoader
from saldet.dataset import SaliencyDataset
from typing import List, Dict, Union, Callable, Optional

def create_dataloader(
    root_dir: str,
    train: bool,
    batch_size: int,
    transform: Callable = None, 
    shuffle: bool = True,
    num_workers: int = 5,
    pin_memory: bool = True,
    drop_last: bool = False,
    persistent_workers: bool = True,
) -> DataLoader:
    """Setup a dataloader for a dataset

    Args:
        root_dir (str): dataset root dir
        train (bool): if True train loader else test/val loader.
        batch_size (int): batch size
        transform (Callable, optional): train data augmentation. Defaults to None.
        shuffle (bool, optional): whether to shuffle dataset. Defaults to True.
        num_workers (int, optional): num workers. Defaults to 5.
        pin_memory (bool, optional): data loader pin memory. Defaults to True.
        drop_last (bool, optional): drop last data loader. Defaults to False.
        persistent_workers (bool, optional): persistent workers data loader. Defaults to True.

    Returns:
        DataLoader: dataset dataloader
    """
    
    dataset = SaliencyDataset(
        root_dir=root_dir,
        train=train,
        transform=transform
    )
    
    if persistent_workers and num_workers==0:
        print(f"> [WARNING] persistent_workers is set to True and num_workers is 0 (must be >0). This is not right. Setting persistent_workers to False.")
        persistent_workers = False
        
    return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle if train else False,
            num_workers=num_workers,
            drop_last=drop_last,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers
        )
    
    
    
    