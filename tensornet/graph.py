import torch
import tensornetwork as tn
import pytorch_lightning as pl
import numpy as np

from tensornet.utils import tensor_tree_node_init
from tensornet.basemodels import SimpleFeedForwardNN
from ivbase.transformers.features.molecules import AdjGraphTransformer, SequenceTransformer
from gnnfp.utils import GraphFPDataset


tn.set_default_backend("pytorch")
torch.set_default_tensor_type(torch.FloatTensor)


class GraphTensorNetwork(pl.LightningModule):

    def __init__(self, 
                dataset,
                max_degree,
                bond_dim,
                edge_structure = 'list'
    ):
        super().__init__()

        self.dataset = dataset
        if max_degree != None:
            self.max_degree = max_degree
        else:
            self.max_degree = self.get_max_degree()

        self.bond_dim = bond_dim
        self.edge_structure = edge_structure

        self.input_dim = dataset.__getitem__(0)[1].shape[1]
        self.tensor_cores = {}
        for i in range(1,max_degree + 1):
            shape = [self.input_dim] + [bond_dim for j in range(i)]
            shape = tuple(shape)
            core = tensor_tree_node_init(shape)
            core = torch.nn.Parameter(core)
            self.tensor_cores[i]=core

    def get_max_degree(self):
        max_degrees_list = torch.zeros(self.dataset.__len__())
        for i in range(self.dataset.__len__()):
            adjacency, features, value = self.dataset.__getitem__(i)
            degrees = torch.sum(adjacency)
            max_degree = torch.max(degrees)
            max_degree_list[i] = max_degree
        return torch.max(max_degree_list)

    def forward(self, edge_data, features):
        nodes, atom_features = self.connect_net(edge_data, features)
        return tn.contractors.auto(nodes+atom_features)
        

    def connect_net(self, edge_data, features):
        #Connects the nodes of the network. Returns the connected core nodes.
        if self.edge_structure == 'adjacency':
            adjacency = edge_data
            edge_indices_x, edge_indices_y = torch.where(adjacency)

        elif self.edge_structure == 'list':
            edge_indices_x, edge_indices_y = edge_data
            edge_indices_x = edge_indices_x.to(dtype=torch.int)
            edge_indices_y = edge_indices_y.to(dtype=torch.int)
            num_nodes = edge_indices_x.shape[0]
            adjacency = torch.zeros(num_nodes,num_nodes)
            for i,j in zip(edge_indices_x, edge_indices_y):
                
                adjacency[i,j] = 1

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

    print('list edge data')
    data_path = os.path.join(os.path.dirname(tensornet.__path__._path[0]), 'data/qm9_80.csv')
    features_path = os.path.join(os.path.dirname(tensornet.__path__._path[0]), 'data/qm9_80/tree.db')
    dataset = GraphFPDataset(data_path, features_path)
    moltennet = GraphTensorNetwork(
        dataset=dataset,
        max_degree=4,
        bond_dim=20,
        edge_structure='list'
        )

    for i in range(4,dataset.__len__()):
        smile, graph, values, ones = dataset.__getitem__([i])
        edge_data = graph.edges()
        result = moltennet(edge_data,features)
        print(result.tensor)

    print('adjacency edge data')
    datapath = os.path.join(os.path.dirname(tensornet.__path__._path[0]), 'data/qm9_80.csv')
    dataset = MolGraphDataset(datapath, scaler=None, smiles_col='smiles')
    moltennet = GraphTensorNetwork(
        dataset=dataset,
        max_degree=4,
        bond_dim=20,
        edge_structure='adjacency'
        )

    for i in range(4,dataset.__len__()):
        adjacency, features, value = dataset.__getitem__([i])
        result = moltennet(adjacency,features)
        print(result.tensor)

    