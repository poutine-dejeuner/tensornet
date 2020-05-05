import os
import torch
import pytorch_lightning
from pytorch_lightning import Trainer
from ivbase.transformers.scaler import StandardScaler

from tensornet.regressor import Regressor
from tensornet.umps import UMPS, MultiUMPS
from tensornet.dataset import MolDataset, CosineDataset
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
    dataset = MolDataset(datapath, scaler=scaler, smiles_col='smiles')

    # datapath = os.path.join(os.path.dirname(tensornet.__path__._path[0]), 'data/cosine_dataset_len-30-30_num-freq-3.csv')
    # dataset = CosineDataset(datapath, scaler=scaler, features_prefix='feat_', labels_prefix=['amplitude_'])
    
    model = UMPS(dataset=dataset, bond_dim = 50, tensor_init='eye',
                input_nn_depth=1, input_nn_out_size=8, batch_max_parallel=4)
    # model = MultiUMPS(dataset=dataset, bond_dim = 50, tensor_init='eye',
    #             input_nn_depth=1, input_nn_out_size=32, output_n_umps=4, output_depth=1)
    
    
    regressor = Regressor(model=model, dataset=dataset, lr=1e-3, batch_size=16,
                validation_split=0.2, random_seed=RANDOM_SEED, num_workers=1, dtype=DTYPE)

    trainer = Trainer(gpus=gpus, min_epochs=20, max_epochs=20)
    trainer.fit(regressor)


