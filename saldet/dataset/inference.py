import os
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from typing import List, Tuple, Callable
from saldet.utils import to_tensor_format
from saldet.io import read_rgb, read_binary
   
class InferenceDataset(Dataset):
    
    EXTENSIONS = (
        "jpg",
        "jpeg",
        "png",
        "ppm",
        "bmp",
        "pgm",
        "tif",
        "tiff",
        "webp",
    )
    
    def __init__(
        self,
        root_dir: str,
        transform: Callable = None,
    ) -> None:
        """Inference Dataset (a folder with images)

        Args:
            root_dir (str): root data dir (must be with train and val folders)
            transform (Callable, optional): set of data transformations. Defaults to None.

        Raises:
            FileNotFoundError: if something is found erroneous in the dataset
        """
        
        super().__init__()
        # checking structure
        try:
            self._sanity_check(root_dir=root_dir)
        except Exception as e:
            raise e
        
        self.data_dir = root_dir
        self.images = [os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir) if f.split(".")[-1].lower() in self.EXTENSIONS]
        self.transform = transform
    
    def _sanity_check(
        self,
        root_dir: str,
    ):
        """Checks dataset structure

        Args:
            root_dir (str): data directory
            
        Raises:
            FileNotFoundError: if the data folder is not right based on the structure in class_map
            FileExistsError: if some label does not have images in its folder

        """
        if not (os.path.exists(root_dir)):
                raise FileNotFoundError(f"Folder {root_dir} does not exist") 
        images = [f for f in os.listdir(root_dir) if f.split(".")[-1].lower() in self.EXTENSIONS]
        if len(images) == 0:
            raise FileExistsError(f"Folder {root_dir} does not have images.")
           
        print(f"> Inference dataset sanity check OK")
         
    def __getitem__(self, index) -> Tuple[torch.Tensor, str]:
        
        image_path = self.images[index]
        image_name = image_path.split(os.sep)[-1]
        image = read_rgb(image_path)
        shape = image.shape
        
        if self.transform is not None:
            image, _ = self.transform(image, None)
            
        image = torch.from_numpy(to_tensor_format(image))
        
        return image, shape, os.path.splitext(image_name)[0]
            
    def __len__(self):
        return len(self.images)