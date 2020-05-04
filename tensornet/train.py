import os
import torch
import pytorch_lightning
from pytorch_lightning import Trainer

from tensornet.regressor import Regressor
from tensornet.umps import UMPS, MultiUMPS
from tensornet.dataset import MolDataset
from tensornet.utils import normalise

import tensornet

if __name__ == "__main__":

    RANDOM_SEED = 42
    torch.random.manual_seed(RANDOM_SEED)

    if torch.cuda.is_available():
        gpus = 1
    else:
        gpus = None

    datapath = os.path.join(os.path.dirname(tensornet.__path__._path[0]), 'data/qm9_mini.csv')
    normaliser = normalise(datapath)
    dataset = MolDataset(datapath, normaliser)
        
    # model = UMPS(feature_dim = 40, bond_dim = 100, output_dim = 19, tensor_init='eye',
    #             input_nn_depth=1, input_nn_out_size=32)
    model = MultiUMPS(feature_dim = 40, bond_dim = 50, output_dim = 19, tensor_init='eye',
                input_nn_depth=1, input_nn_out_size=32, output_n_umps=4, output_depth=1)
    
    
    
    regressor = Regressor(model=model, dataset=dataset, transform=normaliser, lr=1e-4, batch_size=4,
                validation_split=0.2, random_seed=RANDOM_SEED, num_workers=1)

    trainer = Trainer(gpus=gpus, min_epochs=20, max_epochs=20)
    trainer.fit(regressor)


