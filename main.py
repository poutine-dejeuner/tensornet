from pytorch_lightning import Trainer
from argparse import ArgumentParser
from umps import UMPS

"""
For cluster computing.

python main.py --gpus 1

"""

def main(hparams):
    model = UMPS(feature_dim = 41, bond_dim = 100)
    trainer = Trainer(gpus=hparams.gpus)
    trainer.fit(model)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--gpus', default=None)
    args = parser.parse_args()

    main(args)
