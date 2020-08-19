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
        device = 'cuda:0'
    else:
        gpus = None
        device = 'cpu'

    scaler = TorchScalerWrapper(StandardScaler())
    name = 'data/qm9_80'
    file_path = os.path.dirname(tensornet.__path__._path[0])
    data_path = os.path.join(file_path, name + '.csv')
    features_path = os.path.join(file_path, name + '/tree.db')
    cache_file_path = os.path.join(file_path, name + '/cache.hmm')
    cache_file_path = None
    dataset = MolGraphDataset(data_path, features_path, num_labels = 1, scaler = scaler, 
    cache_file_path = cache_file_path)

    model = StaticGraphTensorNetwork(dataset = dataset, 
                                    max_degree = 4, 
                                    max_depth = 4, 
                                    bond_dim = 20, 
                                    embedding_dim = 16,
                                    uniform = True,
                                    device = device)
    
    num_workers = os.cpu_count()
    regressor = MolGraphRegressor(model = model, 
                                dataset = dataset, 
                                loss_fun = torch.nn.MSELoss(reduction='sum'), 
                                lr = 1e-4, batch_size = 16, 
                                validation_split = 0.2, 
                                random_seed = RANDOM_SEED, 
                                num_workers = num_workers, 
                                dtype = DTYPE, 
                                weight_decay = 0)

    #profiler = pl.profiler.AdvancedProfiler()
    profiler = None
    trainer = Trainer(gpus = gpus, min_epochs = 1, max_epochs = 20, profiler = profiler)
    trainer.fit(regressor)
