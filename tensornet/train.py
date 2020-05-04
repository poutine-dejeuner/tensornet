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

    torch.random.manual_seed(111)

    if torch.cuda.is_available():
        gpus = 1
    else:
        gpus = None

    datapath = os.path.join(os.path.dirname(tensornet.__path__._path[0]), 'data/qm9_mini.csv')
    normaliser = normalise(datapath)
    dataset = MolDataset(datapath, normaliser)
        
    model = UMPS(feature_dim = 40, bond_dim = 20, output_dim = 19, tensor_init='eye',
                input_nn_depth=0, input_nn_out_size=32)
    # model = MultiUMPS(feature_dim = 40, bond_dim = 20, output_dim = 19, tensor_init='norm',
    #             input_nn_depth=1, input_nn_out_size=32, output_n_umps=4, output_depth=1)
    
    
    
    regressor = Regressor(model=model, dataset=dataset, transform=normaliser, lr=1e-4, batch_size=4,
                validation_split=0.2, random_seed=42, num_workers=1)

    trainer = Trainer(gpus=gpus, min_epochs=25, max_epochs=50)
    trainer.fit(regressor)


