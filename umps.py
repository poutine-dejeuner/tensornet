import torch
import tensornetwork as tn
import pytorch_lightning as pl

import torch.nn.functional as F
from typing import List
from utils import evaluate_input, batch_node

tn.set_default_backend("pytorch")
torch.set_default_tensor_type(torch.DoubleTensor)

class UMPS(pl.LightningModule):

    def __init__(self, 
                feature_dim: int, 
                bond_dim: int,
                output_dim: int = 0):
        """
        A matrix produt state that has the same core tensor at each nodes. This 
        is an implementation of https://arxiv.org/abs/2003.01039

        Args:
            feature_dim:    The dimension of the embedding of each of the inputs.
            bond_dim:       The dimension of the bond between each cores.
            output_dim:     The dimension of the result of the contraction of the 
                            network.

        Returns:
            contracted_node: Tensor resulting in the contraction of the network.
            If using batch inputs, the result is tensor of dimension 
            (output_dim, batch_dim)
        """

        super().__init__()

        self.feature_dim = feature_dim
        self.output_dim = output_dim
        self.bond_dim = bond_dim
        self.alpha = torch.nn.Parameter(torch.randn(bond_dim, requires_grad = True))
        self.omega = torch.nn.Parameter(torch.randn(bond_dim, requires_grad = True))
        self.output_core = torch.nn.Parameter(torch.randn(bond_dim,output_dim,bond_dim, 
                                                                requires_grad = True))

        #The tensor core of the UMPS is initialized. A second tensor eye is 
        #constructed and concatenated to tensor_core to construct the batch_core.
        #The point of batch core is that when contracted with a padding vector as
        #input the resulting matrix is the identity.
        tensor_core = torch.nn.Parameter(torch.randn(bond_dim,feature_dim,bond_dim, 
                                                                requires_grad = True))
        eye = torch.eye(bond_dim,bond_dim, requires_grad = False)
        batch_core = torch.zeros(bond_dim, 1 + feature_dim, bond_dim)
        batch_core[:, 0, :] = eye
        batch_core[:, 1:, :] = tensor_core
        self.tensor_core = [tensor_core, batch_core]
        self.register_parameter('tensor core', tensor_core)
        self.register_parameter('output core', self.output_core)
        self.register_parameter('alpha vector', self.alpha)
        self.register_parameter('omega vector', self.omega)

    def forward(self, inputs: torch.Tensor):
        """
        Takes a batch input tensor, computes the number of inputs, creates a UMPS
        of length length equal to the number of inputs, connects the input nodes
        to the corresponding tensor nodes and returns the resulting contracted tensor.

        Args:
            inputs:     A torch tensor of dimensions (batch_dim, input_len, feature_dim)
        
        Returns:        A torch tensor of dimensions (batch_dim, output_dim)
        """

        input_len = inputs.size(1)
        has_batch = inputs.size(0) > 1

        #splice the inputs tensor in the input_len dimension
        input_list = [inputs.select(1,i) for i in range(input_len)]
        input_list = [tn.Node(vect) for vect in input_list]
        input_node_list = [tn.Node(self.tensor_core[has_batch]) for i in range(input_len)]

        if self.output_dim > 0:
            output_node = tn.Node(self.output_core, name = 'output') 

            #add output node at the center of the input nodes
            node_list = input_node_list.copy()
            node_list.insert(input_len//2, output_node)
        elif self.output_dim == 0:
            node_list = input_node_list

        #connect tensor cores
        for i in range(len(node_list)-1):
            node_list[i][2]^node_list[i+1][0]

        #connect the alpha and omega nodes to the first and last nodes
        tn.Node(self.alpha, name = 'alpha')[0]^node_list[0][0]
        tn.Node(self.omega, name = 'omega')[0]^node_list[len(node_list)-1][2]

        return evaluate_input(input_node_list, input_list).tensor

    def train_dataloader(self, batch_size):
        num_workers = os.cpu_count()
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                        sampler=train_sampler, num_workers=num_workers )
        return train_loader

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4, weight_decay=0)

    def training_step(self, batch, batch_idx):
        x, y = batch
        predictions = self(x)
        loss = F.mse_loss(predictions,y)
        mae = F.l1_loss(x,y)
        self.logger.summary.scalar('MSE loss', loss)
        self.logger.summary.scalar('MAE', mae)
        return loss

if __name__=='__main__':

    from dataset import MolDataset
    import os

    filedir = os.path.dirname(os.path.realpath(__file__))
    dataset = MolDataset(os.path.join(filedir,'qm9.csv'))
    inputs = dataset.__getitem__(4)
    mps = UMPS(feature_dim = 41, bond_dim = 100)
    print(mps(inputs[0]))