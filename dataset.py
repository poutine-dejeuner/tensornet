import torch, os
import pandas as pd
from torch.utils.data import Dataset
from ivbase.transformers.features.molecules import SequenceTransformer

class MolDataset(Dataset):
    """
    The __getitem__ function takes in a list of indices (of tensor) and returns two 
    tensors parsed_smiles and values. If L is the length of the longest requested smiles,
    and self.vocabulary is the set of symbols the smiles can use, the tensor parsed_smiles 
    has dimensions (batch_size, L, len(self.vocabulary) + 1). The reason for the "+ 1" in 
    the last dimension is that the vector (1,0,...,0) is added as a padding vector for 
    smiles that have shorter length than the max length among batch elements.

    __getitem__
    returns:
        parsed_smiles: tensor of dimension (batch_size, L, len(self.vocabulary) + 1)
        values: tensor of dimension (batch_size, 19)
    """

    def __init__(self, csvpath, transform = None):

        self.path=csvpath
        df=pd.read_csv(self.path)
        self.values =  df.iloc[:,2:].to_numpy()
        self.smiles = df['smiles'].to_numpy().tolist()
        self.dtype=torch.float
        self.vocabulary = ['#', '%', ')', '(', '+', '*', '-', '/', '.',
                           '1', '0', '3', '2', '5', '4', '7', '6', '9', '8', 
                           ':', '=', '@', '[', ']', '\\', 'c', 'o', 'n', 's', 
                           'H', 'B', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl',  'Br', 'I']
        self.seq_transfo = SequenceTransformer(self.vocabulary, True)  
        self.transform = transform      

    def __len__(self):
        return len(self.values)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        values = torch.Tensor(self.values[idx])
        smiles = self.smiles[idx]
        if type(smiles) is str:
            smiles=[smiles]

        #transform the smile in a one-hot vector for self.vocabulary. If the
        #batch smiles are not all the same length, padding vectors (1,0,...,0)
        #are added for each missing caracters.
        parsedsmiles = self.seq_transfo(smiles).type(torch.double)
        if self.transform != None:
            values = self.transform(values)
        
        return parsedsmiles, values

if __name__=='__main__':
    filedirpath = os.path.dirname(os.path.realpath(__file__))
    datapath = os.path.join(filedirpath,'data','qm9_mini.csv')
    ds=MolDataset(datapath)
    item=ds.__getitem__(4)
    print(item[0].size())
    print(item[1].size())
    
    from utils import normalise
    transfo = normalise(datapath)
    item = item[1]
    print(item)
    item = transfo(item)
    print(item)
    print(transfo.inverse(item))
    print('yo')