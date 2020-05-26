import torch
import math

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
        onehot = self.seq_transfo(number).type(self.dtype).squeeze(0)

        transform = transforms.ToTensor()
        image = transform(image).squeeze().unsqueeze(2)
        image = image.type(self.dtype)
        zeros = torch.zeros_like(image)
        cos = torch.cos(image*math.pi/2)
        sin = torch.sin(image*math.pi/2 )
        image = torch.stack((zeros,cos,sin),2).squeeze()
        shape = image.shape
        image = image.reshape(shape[0]*shape[1],3)
        image = image.unsqueeze(0)

        return image, onehot

    def to(self, dtype):
        self.dtype = dtype
        if self.scaler is not None:
            self.scaler.to(dtype)
        return self

if __name__=='__main__':
    
    data = MNIST(dtype = torch.float, root  = './mnist',download=True)
    item=data.__getitem__(0)
    print(item)
    print('sick code bro')
