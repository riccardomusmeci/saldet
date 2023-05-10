<p align="center">
    <img width="100%" src=".static/example_1.png" alt>
</p>

# saldet
**Sal**iency **Det**ection (*saldet*) is a collection of models and tools to perform Saliency Detection with PyTorch.

## **Models**
List of saliency detection models supported by saldet:

* U2Net - https://arxiv.org/abs/2005.09007v3
* PGNet - https://arxiv.org/abs/2204.05041 (follow training instructions from [PGNet's repo](https://github.com/iCVTEAM/PGNet))


## **Train**
### **Easy Mode**
The library comes with easy access to train models thanks to the amazing PyTorch Lightning support. 

```python
from saldet.experiment import train

train(
    data_dir=...,
    config_path="config/u2net_lite.yaml", # check the config folder with some configurations
    output_dir=...,
    resume_from=...,
    seed=42
)
```

Once the training is over, configuration file and checkpoints will be saved into the output dir.

**[WARNING]** The dataset must be structured as follows:
```
dataset
    ├── train                    
    |       ├── images          
    |       │   ├── img_1.jpg
    |       │   └── img_2.jpg                
    |       └── masks
    |           ├── img_1.png
    |           └── img_2.png   
    └── val
           ├── images          
           │   ├── img_10.jpg
           │   └── img_11.jpg                
           └── masks
               ├── img_10.png
               └── img_11.png   
```

### **PyTorch Lighting Mode**
saldet provides utils for model and data PyTorch Lightning Modules.
```python
import pytorch_lightning as pl
from saldet import create_model
from saldet.pl import SaliencyPLDataModule, SaliencyPLModel
from saldet.transform import SaliencyTransform

# datamodule
datamodule = SaliencyPLDataModule(
    root_dir=data_dir,
    train_transform=SaliencyTransform(train=True, **config["transform"]),
    val_transform=SaliencyTransform(train=False, **config["transform"]),
    **config["datamodule"],
)

model = create_model(...)
criterion = ...
optimizer = ...
lr_scheduler = ...

pl_model = SaliencyPLModel(
    model=model, criterion=criterion, optimizer=optimizer, lr_scheduler=lr_scheduler
)

trainer = pl.Trainer(...)

# fit
print(f"Launching training...")
trainer.fit(model=pl_model, datamodule=datamodule)
```

### **PyTorch Mode**
Alternatively you can define your custom training process and use the ```create_model()``` util to use the model you like.
