import torch, os
from torchvision import transforms
import tensornet
import pytorch_lightning as pl

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from ivbase.transformers.scaler import StandardScaler

from tensornet.regressor import ClassifyRegressor
from tensornet.umps import UMPS, MultiUMPS

import dataset



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

    dataset = dataset.MNIST(dtype = DTYPE, root = './mnist', download=True)
    #dataset = torch.utils.data.Subset(dataset, list(range(100)))
    
    model = UMPS(
                dataset=dataset, 
                bond_dim = 20, 
                tensor_init='eye',
                input_nn_depth=0, 
                input_nn_out_size=8, 
                batch_max_parallel=4, 
                device=device)
    
    num_workers = os.cpu_count()
    regressor = ClassifyRegressor(
                model=model, 
                dataset=dataset, 
                loss_fun = torch.nn.functional.cross_entropy, 
                lr=1e-3, 
                batch_size=16, 
                validation_split=0.2, 
                random_seed=RANDOM_SEED, 
                num_workers=num_workers, 
                dtype=DTYPE)

    profiler = pl.profiler.AdvancedProfiler()
    trainer = Trainer(
                gpus=gpus, 
                min_epochs=1, 
                max_epochs=20, 
                profiler=profiler)
    #checkpoint_callback = ModelCheckpoint(filepath='/models/{epoch}-{val_loss:.2f}-{batch_size:.2f}',
    #        monitor='accuracy', save_top_k=5)
    trainer.fit(regressor)


