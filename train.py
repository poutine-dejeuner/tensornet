import os
import torch
import pytorch_lightning
from pytorch_lightning import Trainer
from umps import UMPS
from dataset import MolDataset



if __name__ == "__main__":

    torch.random.manual_seed(111)

    if torch.cuda.is_available():
        gpus = 1
    else:
        gpus = None
    model = UMPS(feature_dim = 40, output_dim = 19, bond_dim = 50)
    # filedir = os.path.dirname(os.path.realpath(__file__))
    # dataset = MolDataset(os.path.join(filedir, 'data/qm9.csv'))
    trainer = Trainer(gpus=gpus, min_epochs=10, max_epochs=20)
    trainer.fit(model)


