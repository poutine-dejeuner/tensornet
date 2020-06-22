import unittest as ut
import tensornet
import os
from tensornet.utils import tensor_tree_node_init
from tensornet.dataset import MolGraphDataset
from tensornet.graph import MoleculeTensorNetwork

class Test_MoleculeTensorNetwork(ut.TestCase):
    
    def test_connect_net_result_shape(self):
        datapath = os.path.join(os.path.dirname(tensornet.__path__._path[0]), 'data/qm9_80.csv')
        dataset = MolGraphDataset(datapath, scaler=None, smiles_col='smiles')
        networker = MoleculeTensorNetwork(
            dataset=dataset,
            max_degree=4,
            bond_dim=100
            )
        adjacency, features, value = dataset.__getitem__([8])
        result = networker(adjacency,features).tensor
        self.assertEqual(len(result.shape), 0)
        
        
        
if __name__ == '__main__':
    ut.main()