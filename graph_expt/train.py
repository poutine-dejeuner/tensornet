import os
import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from ivbase.transformers.scaler import StandardScaler

from tensornet.regressor import Regressor
from tensornet.umps import UMPS, MultiUMPS
from tensornet.graph import GraphTensorNetwork
from tensornet.dataset import MolDataset, CosineDataset
from tensornet.utils import TorchScalerWrapper
from tensornet.basemodels import LSTMPredictor

import gnnfp
from gnnfp.gnnfp import *
from gnnfp.utils import *


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

    file_path = os.path.dirname(tensornet.__path__[0])
    data_path = os.path.join(file_path, 'data/qm9_mini.csv')
    features_path = os.path.join(file_path, 'data/qm9_mini/tree.db')
    dataset = GraphFPDataset(data_path, features_path)

    # datapath = os.path.join(os.path.dirname(tensornet.__path__._path[0]), 'data/cosine_dataset_len-30-30_num-freq-3.csv')
    # dataset = CosineDataset(datapath, scaler=scaler, features_prefix='feat_', labels_prefix=['amplitude_', 'wavelength_'])

    model = GraphTensorNetwork()
    #model = MultiUMPS(dataset=dataset, bond_dim = 50, tensor_init='eye',
    #             input_nn_depth=1, input_nn_out_size=32, output_n_umps=16, output_depth=1)
    
    #model = LSTMPredictor(dataset=dataset, 
    #                lstm_depth=2, lstm_hidden_size=100, lstm_bidirectional=True, lstm_kwargs=None, 
    #                out_nn_depth=1, out_nn_kwargs=None)
    
    # model = LSTMPredictor(dataset=dataset, 
    #                 lstm_depth=2, lstm_hidden_size=100, lstm_bidirectional=True, lstm_kwargs=None, 
    #                 out_nn_depth=1, out_nn_kwargs=None)
    num_workers = os.cpu_count()
    regressor = MolGraphRegressor(model=model, dataset=dataset, loss_fun = torch.nn.MSELoss(reduction='sum'), 
                lr=1e-3, batch_size=16, validation_split=0.2, random_seed=RANDOM_SEED, 
                num_workers=num_workers, dtype=DTYPE, weight_decay=1)

    profiler = pl.profiler.AdvancedProfiler()
    trainer = Trainer(gpus=gpus, min_epochs=1, max_epochs=6, profiler=profiler)
    trainer.fit(regressor)


