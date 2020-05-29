import os, torch
import unittest as ut
import tensornet

from tensornet.dataset import MolDataset
from tensornet.umps import UMPS

class Test_UMPS(ut.TestCase):
    def test_tensor_cat(self):
        datapath = os.path.join(os.path.dirname(tensornet.__path__._path[0]), 'data/qm9_80.csv')
        dataset = MolDataset(datapath, smiles_col='smiles')
        model = UMPS(dataset=dataset, bond_dim = 100, tensor_init='eye',
                    input_nn_depth=0, input_nn_out_size=32, batch_max_parallel=4)

        batch_core = torch.cat((model.eye, model.tensor_core), 1)
        onehot, values = dataset.__getitem__([1])
        onehot = onehot.squeeze()
        v1 = torch.einsum('i,j,ijk->k',model.alpha, onehot, batch_core)
        v2 = torch.einsum('i,j,ijk->k',model.alpha, onehot[1:], batch_core[:,1:,:])
        self.assertTrue(torch.equal(v1,v2))
        v3 = torch.zeros_like(onehot)
        v3[0] = 1
        v4 = torch.einsum('i,j,ijk->k',model.alpha, v3, batch_core)
        self.assertTrue(torch.equal(v4,model.alpha))

    def test_output_node(self):
        '''checks that the output node is always at the same place for a given input over 
        different batches
        '''
        return

if __name__ == '__main__':    
    ut.main()
