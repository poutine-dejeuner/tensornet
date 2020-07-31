import torch, os
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from ivbase.transformers.scaler import StandardScaler

from tensornet.regressor import MolGraphRegressor
from tensornet.graph import StaticGraphTensorNetwork, GraphTensorNetwork, MolGraphDataset
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

    file_path = os.path.dirname(tensornet.__path__._path[0])
    data_path = os.path.join(file_path, 'data/qm9_mini.csv')
    features_path = os.path.join(file_path, 'data/qm9_mini/tree.db')
    dataset = MolGraphDataset(data_path, features_path, values_column = 0)

    model = StaticGraphTensorNetwork(dataset = dataset, max_degree = 4, max_depth = 4, bond_dim = 2)
    
    num_workers = 0 #os.cpu_count()
    regressor = MolGraphRegressor(model=model, dataset=dataset, loss_fun = torch.nn.MSELoss(reduction='sum'), 
                lr=1e-3, batch_size=4, validation_split=0.2, random_seed=RANDOM_SEED, 
                num_workers=num_workers, dtype=DTYPE, weight_decay=1)

    profiler = pl.profiler.AdvancedProfiler()
    trainer = Trainer(gpus=gpus, min_epochs=1, max_epochs=6, profiler=profiler)
    trainer.fit(regressor)
