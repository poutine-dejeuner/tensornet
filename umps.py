import os
import torch
import numpy as np
import tensornetwork as tn
import pytorch_lightning as pl
from dataset import MolDataset

import torch.nn.functional as F
from typing import List
from utils import evaluate_input, batch_node, tensor_norm, create_tensor, normalise
from ivbase.nn.base import FCLayer

tn.set_default_backend("pytorch")
torch.set_default_tensor_type(torch.DoubleTensor)


class UMPS(pl.LightningModule):

    def __init__(self, 
                feature_dim:int, 
                bond_dim:int,
                output_dim:int=0,
                tensor_init='eye',
                input_nn_depth:int=0,
                input_nn_out_size:int=32,
                input_nn_kwargs=None):
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

        self._unused = torch.nn.LSTM(5, 5) # this LSTM doesnt do anything, only avoids a bud in pytorch-lightning

        # Basic attributes
        self.feature_dim = feature_dim
        self.output_dim = output_dim
        self.bond_dim = bond_dim
        self.input_nn_depth = input_nn_depth
        self.input_nn_out_size = input_nn_out_size
        self.input_nn_kwargs = input_nn_kwargs
        
        #The tensor core of the UMPS is initialized. A second tensor eye is 
        #constructed and concatenated to tensor_core to construct the batch_core.
        #The point of batch core is that when contracted with a padding vector as
        #input the resulting matrix is the identity.
        tensor_num_feats = input_nn_out_size if input_nn_depth > 0 else self.feature_dim + 1
        tensor_core = create_tensor((bond_dim, tensor_num_feats - 1, bond_dim), requires_grad=True, opt=tensor_init)
        eye = torch.eye(bond_dim,bond_dim, requires_grad = False)
        batch_core = torch.zeros(bond_dim, tensor_num_feats, bond_dim)
        batch_core[:, 0, :] = eye
        batch_core[:, 1:, :] = tensor_core[:, :, :]
        self.tensor_core = torch.nn.Parameter(batch_core)

        # Initializing other tensors for the tensor network
        self.alpha = create_tensor((bond_dim), requires_grad=True, opt='norm')
        self.alpha = torch.nn.Parameter(self.alpha)
        self.omega = create_tensor((bond_dim), requires_grad=True, opt='norm')
        self.omega = torch.nn.Parameter(self.omega)
        self.output_core = create_tensor((bond_dim, output_dim, bond_dim), 
                                            requires_grad=True, opt='norm')
        self.output_core = torch.nn.Parameter(self.output_core)

        # Initializing neural network layers for the inputs
        input_nn_kwargs = {} if input_nn_kwargs is None else input_nn_kwargs
        in_size = self.feature_dim + 1
        if input_nn_depth == 0:
            self.fc_input_layers = []
        elif input_nn_depth == 1:
            input_nn_kwargs['activation'] = 'none'
            self.fc_input_layers = [FCLayer(in_size=in_size, out_size=input_nn_out_size, **input_nn_kwargs)]
        elif input_nn_depth >= 2:
            self.fc_input_layers = [FCLayer(in_size=in_size, out_size=input_nn_out_size, **input_nn_kwargs)]
            fc_input_layers_ext = [FCLayer(in_size=input_nn_out_size, out_size=input_nn_out_size, **input_nn_kwargs) for ii in range(input_nn_depth - 1)]
            self.fc_input_layers.extend(fc_input_layers_ext)
            input_nn_kwargs['activation'] = 'none'
            self.fc_input_layers.append(FCLayer(in_size=input_nn_out_size, out_size=input_nn_out_size, **input_nn_kwargs))
        else:
            raise ValueError('`input_nn_kwargs` must be a positive integer')
        

    def _collate_with_padding(self, inputs):
        tensors_x, tensors_y = zip(*inputs)

        # Generate input feature tensor, with only empty strings denoted by the one-hot vector [1, 0, 0, ...., 0]
        max_input_len = max([tensor.shape[1] for tensor in tensors_x])
        collated_tensors_x = torch.zeros((len(tensors_x), max_input_len, tensors_x[0].shape[2]), 
                                                                dtype=tensors_x[0].dtype)
        collated_tensors_x[:, :, 0] = 1
        
        # Replace the non-empty one-hot by the input tensor_x
        for ii, tensor in enumerate(tensors_x):
            collated_tensors_x[ii, :tensor.shape[1], :] = tensor

        # Stack the input tensor_y
        collated_tensors_y = torch.stack(tensors_y, dim=0)

        return collated_tensors_x, collated_tensors_y


    def forward(self, inputs: torch.Tensor):
        """
        Takes a batch input tensor, computes the number of inputs, creates a UMPS
        of length length equal to the number of inputs, connects the input nodes
        to the corresponding tensor nodes and returns the resulting contracted tensor.

        Args:
            inputs:     A torch tensor of dimensions (batch_dim, input_len, feature_dim)
        
        Returns:        A torch tensor of dimensions (batch_dim, output_dim)
        """

        # 
        for fc_layer in self.fc_input_layers:
            inputs = fc_layer(inputs)
        # if len(self.fc_input_layers) > 0:
        #     inputs = F.softmax(inputs, dim=-1)
        
        #splice the inputs tensor in the input_len dimension
        input_len = inputs.size(1)
        input_list = [inputs.select(1,i) for i in range(input_len)]
        input_list = [tn.Node(vect) for vect in input_list]
        input_node_list = [tn.Node(self.tensor_core) for i in range(input_len)]

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

        output = evaluate_input(input_node_list, input_list).tensor

        return output

    def prepare_data(self):
        filedir = os.path.dirname(os.path.realpath(__file__))
        self.dataset = MolDataset(os.path.join(filedir, 'data/qm9_mini.csv'))
        self.batch_size = 4
        validation_split = .2
        random_seed = 42

        # Creating data indices for training and validation splits:
        dataset_size = len(self.dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(validation_split * dataset_size))
        np.random.seed(random_seed)
        np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]

        # Creating data samplers and loaders:
        self.train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
        valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_indices)

    def train_dataloader(self):
        num_workers = 1 #os.cpu_count()
        train_loader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, 
                num_workers=num_workers, shuffle=True, collate_fn=self._collate_with_padding)
        return train_loader

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4, weight_decay=0)

    def training_step(self, batch, batch_idx):
        x, y = batch
        predictions = self(x)
        loss = F.mse_loss(predictions,y)
        mae = F.l1_loss(predictions,y)
        tensorboard_logs = {'train MSE loss': loss, 'train MAE': mae}
        return {'loss': loss, 'log': tensorboard_logs}



class BaseRegression(pl.LightningModule):

    def __init__(self, batch_size=1, validation_split=0.2, random_seed=42):
        """
        Abstract class
        """

        super().__init__()
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.random_seed = random_seed
        self._unused = torch.nn.LSTM(1, 1) # this LSTM doesnt do anything, only avoids a bud in pytorch-lightning
        

    def _collate_with_padding(self, inputs):
        tensors_x, tensors_y = zip(*inputs)

        # Generate input feature tensor, with only empty strings denoted by the one-hot vector [1, 0, 0, ...., 0]
        max_input_len = max([tensor.shape[1] for tensor in tensors_x])
        collated_tensors_x = torch.zeros((len(tensors_x), max_input_len, tensors_x[0].shape[2]), dtype=tensors_x[0].dtype)
        collated_tensors_x[:, :, 0] = 1
        
        # Replace the non-empty one-hot by the input tensor_x
        for ii, tensor in enumerate(tensors_x):
            collated_tensors_x[ii, :tensor.shape[1], :] = tensor

        # Stack the input tensor_y
        collated_tensors_y = torch.stack(tensors_y, dim=0)

        return collated_tensors_x, collated_tensors_y


    def forward(self, inputs: torch.Tensor):
        """
        Takes a batch input tensor, computes the number of inputs, creates a UMPS
        of length length equal to the number of inputs, connects the input nodes
        to the corresponding tensor nodes and returns the resulting contracted tensor.

        Args:
            inputs:     A torch tensor of dimensions (batch_dim, input_len, feature_dim)
        
        Returns:        A torch tensor of dimensions (batch_dim, output_dim)
        """

        # 
        for fc_layer in self.fc_input_layers:
            inputs = fc_layer(inputs)
        
        #splice the inputs tensor in the input_len dimension
        input_len = inputs.size(1)
        input_list = [inputs.select(1,i) for i in range(input_len)]
        input_list = [tn.Node(vect) for vect in input_list]
        input_node_list = [tn.Node(self.tensor_core) for i in range(input_len)]

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

        output = evaluate_input(input_node_list, input_list).tensor

        return output

    def prepare_data(self):
        filedir = os.path.dirname(os.path.realpath(__file__))
        datapath = os.path.join(filedir, 'data/qm9_mini.csv')
        self.normalise = normalise(datapath)
        self.dataset = MolDataset(datapath)
        self.batch_size = 1
        validation_split = .2
        random_seed = 42

        # Creating data indices for training and validation splits:
        dataset_size = len(self.dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(validation_split * dataset_size))
        np.random.seed(random_seed)
        np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]

        # Creating data samplers and loaders:
        self.train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
        self.valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_indices)

    def train_dataloader(self):
        num_workers = os.cpu_count()
        train_loader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, 
                num_workers=num_workers, sampler=self.train_sampler, 
                collate_fn=self._collate_with_padding)
        return train_loader

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4, weight_decay=0)

    def training_step(self, batch, batch_idx):
        x, y = batch
        predictions = self.normalise.inverse(self(x))
        loss = F.mse_loss(predictions,y)
        mae = F.l1_loss(predictions,y)
        tensorboard_logs = {'train_MSE_loss': loss, 'train_MAE': mae}
        return {'loss': loss, 'log': tensorboard_logs}

    def val_dataloader(self):
        num_workers = os.cpu_count()
        validation_loader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, 
                num_workers=num_workers, sampler=self.valid_sampler,
                collate_fn=self._collate_with_padding)
        return validation_loader

    def validation_step(self, batch, batch_idx):
        x, y = batch
        predictions = self.normalise.inverse(self(x))
        loss = F.mse_loss(predictions,y)
        mae = F.l1_loss(predictions,y)
        tensorboard_logs = {'val_MSE_loss': loss, 'val_MAE': mae}
        return {'val_MSE_loss': loss, 'log': tensorboard_logs}

if __name__=='__main__':

    filedir = os.path.dirname(os.path.realpath(__file__))
    dataset = MolDataset(os.path.join(filedir,'data','qm9.csv'))
    inputs = dataset.__getitem__(4)
    mps = UMPS(feature_dim = 40, bond_dim = 100)
    print(mps(inputs[0]))

