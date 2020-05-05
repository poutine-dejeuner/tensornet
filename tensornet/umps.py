import os
import torch
from torch import nn
import torch.nn.functional as F

import numpy as np
import tensornetwork as tn
import pytorch_lightning as pl

from typing import List
from ivbase.nn.base import FCLayer

from tensornet.utils import evaluate_input, batch_node, tensor_norm, create_tensor


tn.set_default_backend("pytorch")
torch.set_default_tensor_type(torch.DoubleTensor)


class UMPS(nn.Module):

    def __init__(self, 
                bond_dim:int,
                dataset,
                tensor_init='eye',
                input_nn_depth:int=0,
                input_nn_out_size:int=32,
                input_nn_kwargs=None,
                dtype=torch.float,
                batch_max_parallel=4):
        """
        A matrix produt state that has the same core tensor at each nodes. This 
        is an implementation of https://arxiv.org/abs/2003.01039

        Args:
            bond_dim:       The dimension of the bond between each cores.

        Returns:
            contracted_node: Tensor resulting in the contraction of the network.
            If using batch inputs, the result is tensor of dimension 
            (output_dim, batch_dim)
        """

        super().__init__()

        # Basic attributes
        self.dataset = dataset
        self.bond_dim = bond_dim
        self.input_nn_depth = input_nn_depth
        self.input_nn_out_size = input_nn_out_size
        self.input_nn_kwargs = input_nn_kwargs
        self.dtype = dtype
        self.batch_max_parallel = batch_max_parallel

        self.feature_dim = dataset[0][0].shape[-1] - 1
        self.output_dim = dataset[0][1].shape[-1]

        #The tensor core of the UMPS is initialized. An identity matrix is
        #constructed and concatenated to tensor_core to construct the batch_core.
        #The point of batch core is that when contracted with a padding vector as
        #input the resulting matrix is the identity.
        tensor_num_feats = input_nn_out_size if input_nn_depth > 0 else self.feature_dim 
        tensor_core = create_tensor((bond_dim, tensor_num_feats, bond_dim), requires_grad=True, 
                                                                                    opt=tensor_init)
        eye = torch.eye(bond_dim,bond_dim, requires_grad = False, dtype=torch.float)
        batch_core = torch.zeros(bond_dim, tensor_num_feats + 1, bond_dim, dtype=torch.float)
        batch_core[:, 0, :] = eye
        batch_core[:, 1:, :] = tensor_core[:, :, :]
        self.tensor_core = torch.nn.Parameter(batch_core)

        # Initializing other tensors for the tensor network
        self.alpha = create_tensor((bond_dim), requires_grad=True, opt='norm')
        self.alpha = torch.nn.Parameter(self.alpha)
        self.omega = create_tensor((bond_dim), requires_grad=True, opt='norm')
        self.omega = torch.nn.Parameter(self.omega)
        self.output_core = create_tensor((bond_dim, self.output_dim, bond_dim), 
                                            requires_grad=True, opt='norm')
        self.output_core = torch.nn.Parameter(self.output_core)

        self.softmax_temperature = torch.nn.Parameter(torch.Tensor([10.0]).float())

        # Initializing neural network layers for the inputs
        input_nn_kwargs = {} if input_nn_kwargs is None else input_nn_kwargs
        in_size = self.feature_dim
        self.fc_input_layers = nn.ModuleList()
        if input_nn_depth == 0:
            pass
        elif input_nn_depth == 1:
            input_nn_kwargs['activation'] = 'none'
            self.fc_input_layers.append(FCLayer(in_size=in_size, out_size=input_nn_out_size, 
                                                                bias=False, **input_nn_kwargs))
        elif input_nn_depth >= 2:
            self.fc_input_layers.append(FCLayer(in_size=in_size, out_size=input_nn_out_size, 
                                                                bias=False, **input_nn_kwargs))
            fc_input_layers_ext = [FCLayer(in_size=input_nn_out_size, out_size=input_nn_out_size, 
                                bias=False, **input_nn_kwargs) for ii in range(input_nn_depth - 1)]
            self.fc_input_layers.extend(fc_input_layers_ext)
            input_nn_kwargs['activation'] = 'none'
            self.fc_input_layers.append(FCLayer(in_size=input_nn_out_size, out_size=input_nn_out_size, 
                                                                bias=False, **input_nn_kwargs))
        else:
            raise ValueError('`input_nn_depth` must be a positive integer')

        self.to(dtype)
        pass
        

    def forward(self, inputs: torch.Tensor):
        """
        Takes a batch input tensor, computes the number of inputs, creates a UMPS
        of length length equal to the number of inputs, connects the input nodes
        to the corresponding tensor nodes and returns the resulting contracted tensor.

        Args:
            inputs:     A torch tensor of dimensions (batch_dim, input_len, feature_dim)
        
        Returns:        A torch tensor of dimensions (batch_dim, output_dim)
        """
        
        factor = self.batch_max_parallel

        output = [self._forward(inputs[ii:ii+factor]) for ii in range(0, inputs.shape[0], factor)]
        output = torch.cat(output, dim=0)

        return output
        

    def _forward(self, inputs: torch.Tensor):
        #The slice inputs[:,:,0] has 0 for normal inputs and 1 for padding vectors.
        #We need the FC nn to preserve this.
        nned_inputs = inputs[:,:,1:]
        for fc_layer in self.fc_input_layers:
            nned_inputs = fc_layer(nned_inputs)
        d1,d2,d3 = nned_inputs.shape
        new_inputs = torch.zeros(d1,d2,d3+1, dtype=self.dtype)
        new_inputs[:,:,0] = inputs[:,:,0]
        new_inputs[:,:,1:] = nned_inputs
        inputs = new_inputs
        # inputs = F.layer_norm(inputs, inputs.shape[-1:])
        if len(self.fc_input_layers) > 0:
            inputs = F.softmax(inputs*self.softmax_temperature, dim=-1)
        
        #slice the inputs tensor in the input_len dimension
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

        output = evaluate_input(input_node_list, input_list, dtype=self.dtype).tensor

        return output

    def to(self, dtype):
        super().to(dtype)
        self.dtype = dtype
        return self



class MultiUMPS(nn.Module):

    def __init__(self, 
                dataset, 
                bond_dim:int,
                tensor_init='eye',
                input_nn_depth:int=0,
                input_nn_out_size:int=32,
                input_nn_kwargs=None,
                output_n_umps:int=4,
                output_depth:int=1,
                output_nn_kwargs=None,
                batch_max_parallel=4):
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

        # Basic attributes
        self.output_n_umps = output_n_umps
        self.output_depth = output_depth

        # Initializing neural network layers for the outputs
        output_nn_kwargs = {} if output_nn_kwargs is None else output_nn_kwargs
        
        nn_size = output_dim * output_n_umps
        
        if output_depth == 0:
            self.fc_output_layers = []
        elif output_depth == 1:
            output_nn_kwargs['activation'] = 'none'
            self.fc_output_layers = [FCLayer(in_size=nn_size, out_size=output_dim, **output_nn_kwargs)]
        elif output_depth >= 2:
            self.fc_output_layers = [FCLayer(in_size=nn_size, out_size=nn_size, **output_nn_kwargs) for ii in range(input_nn_depth - 1)]
            output_nn_kwargs['activation'] = 'none'
            self.fc_output_layers.append(FCLayer(in_size=nn_size, out_size=output_dim, **output_nn_kwargs))
        else:
            raise ValueError('`input_nn_depth` must be a positive integer')
        self.fc_output_layers = nn.ModuleList(self.fc_output_layers)
        
        # initializing tensor networks
        self.umps = nn.ModuleList([UMPS(
                dataset=dataset, 
                bond_dim=bond_dim,
                tensor_init=tensor_init,
                input_nn_depth=input_nn_depth,
                input_nn_out_size=input_nn_out_size,
                input_nn_kwargs=input_nn_kwargs,
                batch_max_parallel=batch_max_parallel) for _ in range(output_n_umps)])


    def forward(self, inputs: torch.Tensor):
        """
        Takes a batch input tensor, computes the number of inputs, creates a UMPS
        of length length equal to the number of inputs, connects the input nodes
        to the corresponding tensor nodes and returns the resulting contracted tensor.

        Args:
            inputs:     A torch tensor of dimensions (batch_dim, input_len, feature_dim)
        
        Returns:        A torch tensor of dimensions (batch_dim, output_dim)
        """

        umps_results = torch.cat([this_umps(inputs) for this_umps in self.umps], dim=-1)

        output = umps_results
        for fc_layer in self.fc_output_layers:
            output = fc_layer(output)
        
        return output

    def to(self, dtype):
        super().to(dtype)
        for net in self.umps:
            net.to(dtype)
        
        return self

