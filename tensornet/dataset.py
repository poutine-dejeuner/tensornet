import torch, os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from ivbase.transformers.features.molecules import SequenceTransformer, AdjGraphTransformer

from tensornet.arg_checker import check_arg_iterator

class MolGraphDataset(Dataset):

    def __init__(self, csvpath, scaler=None, smiles_col=None, dtype=torch.float, column=0):

        #self.alphabet = ['H', 'B', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl',  'Br', 'I']
        self.path=csvpath
        self.dtype = dtype
        df = pd.read_csv(self.path)
        cols = df.columns

        # Get the smiles
        if smiles_col is None:
            smiles_col = [col for col in cols if (isinstance(col, str) and ('smile' in col.lower()))]
        smiles_col = check_arg_iterator(smiles_col, enforce_type=list)
        if len(smiles_col) != 1:
            raise ValueError(f'There must be a single SMILES column. Provided: {smiles_col}')
        self.smiles = df[smiles_col].values.flatten()

        self.values = df.iloc[:, 2:].values[column]

    def __len__(self):
        return len(self.values)

    def __getitem__(self, idx):
        idx = check_arg_iterator(idx, enforce_type=list)
        smile = self.smiles[idx]
        value = self.values[idx]

        adjtransfo = AdjGraphTransformer(explicit_H=True)

        adjacency, features = adjtransfo(smile)[0][0]

        return adjacency, features, value


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

    def __init__(self, csvpath, scaler=None, smiles_col=None, dtype=torch.float):

        self.path=csvpath
        self.dtype = dtype
        df = pd.read_csv(self.path)
        cols = df.columns

        # Get the smiles
        if smiles_col is None:
            smiles_col = [col for col in cols if (isinstance(col, str) and ('smile' in col.lower()))]
        smiles_col = check_arg_iterator(smiles_col, enforce_type=list)
        if len(smiles_col) != 1:
            raise ValueError(f'There must be a single SMILES column. Provided: {smiles_col}')
        self.smiles = df[smiles_col].values.flatten()

        # Get the values
        #columns = [0,1]
        #self.values = df.iloc[:, 2:].values[:,columns]
        self.values = df.iloc[:, 2:].values
        
        self.vocabulary = ['#', '%', ')', '(', '+', '*', '-', '/', '.',
                           '1', '0', '3', '2', '5', '4', '7', '6', '9', '8', 
                           ':', '=', '@', '[', ']', '\\', 'c', 'o', 'n', 's', 
                           'H', 'B', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl',  'Br', 'I']
        self.seq_transfo = SequenceTransformer(self.vocabulary, True)

        self.scaler = None
        if scaler is not None:
            _, values = self[:]
            self.scaler = scaler.fit(values)


    def __len__(self):
        return len(self.values)

    def __getitem__(self, idx):
        idx = check_arg_iterator(idx, enforce_type=list)

        values = torch.Tensor(self.values[idx])
        smiles = self.smiles[idx]
        if type(smiles) is str:
            smiles=[smiles]

        #transform the smile in a one-hot vector for self.vocabulary. If the
        #batch smiles are not all the same length, padding vectors (1,0,...,0)
        #are added for each missing caracters.
        parsedsmiles = self.seq_transfo(smiles).type(self.dtype)

        if self.scaler != None:
            values = self.scaler.transform(values)
            if isinstance(values, np.ndarray):
                values = torch.Tensor(values)
        values = values.to(self.dtype)

        return parsedsmiles, values

    def to(self, dtype):
        self.dtype = dtype
        if self.scaler is not None:
            self.scaler.to(dtype)
        return self



class CosineDataset(Dataset):
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

    def __init__(self, csvpath, scaler=None, features_prefix='feat_', labels_prefix=['amplitude_', 'wavelength_'], dtype=torch.float):

        self.dtype = dtype
        labels_prefix = check_arg_iterator(labels_prefix)

        self.path=csvpath
        df = pd.read_csv(self.path)
        cols = df.columns

        features_prefix = check_arg_iterator(features_prefix)
        features_cols = [col for col in cols for pref in features_prefix if (isinstance(col, str) and col.startswith(pref))]
        self.features = df[features_cols].values

        labels_prefix = check_arg_iterator(labels_prefix)
        values_cols = [col for col in cols for pref in labels_prefix if (isinstance(col, str) and col.startswith(pref))]
        self.values = df[values_cols].values

        self.scaler = None
        if scaler is not None:
            _, values = self[:]
            self.scaler = scaler.fit(values)
        

    def __len__(self):
        return len(self.values)

    def __getitem__(self, idx):
        
        idx = check_arg_iterator(idx, enforce_type=list)
        
        values = torch.Tensor(self.values[idx])
        features_temp = torch.Tensor(self.features[idx]).to(self.dtype).unsqueeze(-1)
        features = torch.zeros((features_temp.shape[0], self.features.shape[1], 2), dtype=self.dtype)
        features[:, :, 1:] = features_temp
        
        
        if self.scaler != None:
            values = self.scaler.transform(values)
            if isinstance(values, np.ndarray):
                values = torch.Tensor(values)
            
        values = values.to(self.dtype)

        return features, values


    def to(self, dtype):
        self.dtype = dtype
        if self.scaler is not None:
            self.scaler.to(dtype)
        return self

if __name__=='__main__':
    import tensornet
    datapath = os.path.join(os.path.dirname(tensornet.__path__._path[0]), 'data/qm9_80.csv')
    '''
    dataset = MolDataset(datapath, smiles_col='smiles')
    onehot,values = dataset.__getitem__([0])
    print(onehot)
    print(values)

    print(dataset.seq_transfo(['#']))
    print(dataset.seq_transfo(['BI']))
    '''

    dataset = MolGraphDataset(datapath, smiles_col='smiles')
    onehot, features, values = dataset.__getitem__([8])
    print(dataset.__len__())
    print(onehot)
    print(features)
    print(values)
