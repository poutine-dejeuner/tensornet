import os
import torch
import pytorch_lightning
from pytorch_lightning import Trainer

from tensornet.regressor import Regressor
from tensornet.umps import UMPS, MultiUMPS
from tensornet.dataset import MolDataset
from tensornet.utils import normalise


if __name__ == "__main__":

    torch.random.manual_seed(111)

    if torch.cuda.is_available():
        gpus = 1
    else:
        gpus = None
        
    # model = UMPS(feature_dim = 40, bond_dim = 40, output_dim = 19, tensor_init='eye',
    #             input_nn_depth=0, input_nn_out_size=32)
    model = MultiUMPS(feature_dim = 40, bond_dim = 20, output_dim = 19, tensor_init='eye',
                input_nn_depth=0, input_nn_out_size=32, output_n_umps=4, output_depth=1)
    
    filedir = os.path.dirname(os.path.realpath(__file__))
    datapath = os.path.join(filedir, 'data/qm9_mini.csv')
    normaliser = normalise(datapath)
    dataset = MolDataset(datapath, normaliser)
    
    regressor = Regressor(model=model, dataset=dataset, transform=normaliser, lr=1e-4, batch_size=4,
                validation_split=0.2, random_seed=42, num_workers=1)

    trainer = Trainer(gpus=gpus, min_epochs=10, max_epochs=20)
    trainer.fit(regressor)


