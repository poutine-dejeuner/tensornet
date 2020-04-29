import torch
import pytorch_lightning
from pytorch_lightning import Trainer
from umps import UMPS

if torch.cuda.is_available():
    gpus = 1
else:
    gpus = None
model = UMPS(feature_dim = 41, bond_dim = 100)
trainer = Trainer(gpus=gpus)
trainer.fit(model)
