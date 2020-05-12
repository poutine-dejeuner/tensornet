import torch

from torchvision import transforms, datasets
from torch.utils.data import Dataset
from ivbase.transformers.features.molecules import SequenceTransformer

from tensornet.arg_checker import check_arg_iterator

class MNIST(datasets.MNIST):
    def __init__(self, dtype, scaler = None, **kwargs):
        self.scaler = scaler
        self.dtype = dtype
        self.vocabulary = ['0','1','2','3','4','5','6','7','8','9']
        self.seq_transfo = SequenceTransformer(self.vocabulary, True)
        super().__init__(**kwargs)

    def __getitem__(self, idx):
        image, number = super().__getitem__(idx)
        number = [str(number)]
        image = image.type(self.dtype)
        onehot = self.seq_transfo(number).type(self.dtype)
        return image, onehot.squeeze(0)

    def to(self, dtype):
        self.dtype = dtype
        if self.scaler is not None:
            self.scaler.to(dtype)
        return self

if __name__=='__main__':
    transform = transforms.ToTensor()
    data = MNIST(dtype = torch.float, root  = './mnist',download=True, transform=transform)
    print(data.__getitem__(0))
