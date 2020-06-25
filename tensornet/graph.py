import torch
import tensornetwork as tn
import pytorch_lightning as pl
import numpy as np

from tensornet.utils import tensor_tree_node_init
from tensornet.basemodels import SimpleFeedForwardNN
from ivbase.transformers.features.molecules import AdjGraphTransformer, SequenceTransformer


tn.set_default_backend("pytorch")
torch.set_default_tensor_type(torch.FloatTensor)


class MoleculeTensorNetwork(pl.LightningModule):

    def __init__(self, 
                dataset,
                max_degree,
                bond_dim
    ):
        super().__init__()

        self.dataset = dataset
        self.max_degree = max_degree
        self.bond_dim = bond_dim

        self.input_dim = dataset.__getitem__([0])[1].shape[1]
        self.tensor_cores = {}
        for i in range(1,max_degree):
            shape = [self.input_dim] + [bond_dim for j in range(i)]
            shape = tuple(shape)
            core = tensor_tree_node_init(shape)
            core = torch.nn.Parameter(core)
            self.tensor_cores[i]=core

    def forward(self, adjacency, features):
        nodes, atom_features = self.connect_net(adjacency, features)
        return tn.contractors.auto(nodes+atom_features)
        

    def connect_net(self, adjacency, features):
        #Connects the nodes of the network. Returns the connected core nodes.
        edge_indices_x, edge_indices_y = torch.where(adjacency)
        edge_indices_x = edge_indices_x.numpy().tolist()
        edge_indices_y = edge_indices_y.numpy().tolist()
        edges = set()
        for i,j in zip(edge_indices_x, edge_indices_y):
            idx = [i,j]
            idx.sort()
            idx = tuple(idx)
            edges.add(idx)
        degrees = adjacency.sum(axis=1,dtype=torch.int64)
        degrees = degrees.numpy().tolist()
        nodes = []
        for deg in degrees:
            core = self.tensor_cores[deg]
            node = tn.Node(core)
            nodes.append(node)
            
        #connect nodes with inputs
        atom_features = [tn.Node(features.select(dim=0,index=i)) for i in range(features.shape[0])]
        for i, inp in enumerate(atom_features):
            input_edge = inp[0]
            node_edge = nodes[i][0]
            input_edge^node_edge

        #connect nodes
        for edge in edges:
            node0 = nodes[edge[0]]
            node1 = nodes[edge[1]]
            edge0 = node0.get_all_dangling().pop()
            edge1 = node1.get_all_dangling().pop()
            edge0 ^ edge1
        return nodes, atom_features

if __name__ == '__main__':
    import os, tensornet
    from tensornet.dataset import MolGraphDataset

    datapath = os.path.join(os.path.dirname(tensornet.__path__._path[0]), 'data/qm9_80.csv')
    dataset = MolGraphDataset(datapath, scaler=None, smiles_col='smiles')
    moltennet = MoleculeTensorNetwork(
        dataset=dataset,
        max_degree=4,
        bond_dim=100
        )
    print(dataset.__len__())
    for i in range(4,dataset.__len__()):
        adjacency, features, value = dataset.__getitem__([i])
        result = moltennet(adjacency,features)
        print(result.tensor)
