import os
import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from ivbase.transformers.scaler import StandardScaler

from tensornet.regressor import Regressor2
from tensornet.umps import MPS
from tensornet.dataset import MolDataset
from tensornet.utils import TorchScalerWrapper

import tensornet

if __name__ == "__main__":

    DTYPE = torch.float32
    RANDOM_SEED = 42

    torch.random.manual_seed(RANDOM_SEED)
    torch.set_default_dtype(DTYPE)

    if torch.cuda.is_available():
        gpus = 1
    else:
        gpus = None

    scaler = TorchScalerWrapper(StandardScaler())

    datapath = os.path.join(os.path.dirname(tensornet.__path__._path[0]), 'data/qm9_mini.csv')
    dataset = MolDataset(datapath, scaler=scaler, smiles_col='smiles', num_features=1)

    model = MPS(dataset=dataset, bond_dim = 20, tensor_init='eye')
    
    num_workers = 0 #os.cpu_count()
    regressor = Regressor2(model=model, dataset=dataset, loss_fun = torch.nn.MSELoss(reduction='sum'), 
                lr=1e-3, batch_size=16, validation_split=0.2, random_seed=RANDOM_SEED, 
                num_workers=num_workers, dtype=DTYPE, weight_decay=1)

    profiler = pl.profiler.AdvancedProfiler()
    trainer = Trainer(gpus=gpus, min_epochs=1, max_epochs=6, profiler=profiler)
    trainer.fit(regressor)


