import torch, os, pytorch_lightning
from torchvision import transforms
import tensornet

from pytorch_lightning import Trainer
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
    else:
        gpus = None

    transform = transforms.ToTensor()
    dataset = dataset.MNIST(dtype = DTYPE, root = './mnist', download=True, transform=transform)
    
    model = UMPS(dataset=dataset, bond_dim = 20, tensor_init='eye',
                input_nn_depth=1, input_nn_out_size=8, batch_max_parallel=4)
    
    num_workers = os.cpu_count()
    regressor = ClassifyRegressor(model=model, dataset=dataset, loss_fun = torch.nn.functional.cross_entropy, 
                lr=1e-3, batch_size=16, validation_split=0.2, random_seed=RANDOM_SEED, 
                num_workers=num_workers, dtype=DTYPE)

    trainer = Trainer(gpus=gpus, min_epochs=20, max_epochs=20)
    trainer.fit(regressor)


