import os, math, torch
from torch import nn
import torch.nn.functional as F

import numpy as np
import tensornetwork as tn
import pytorch_lightning as pl

from typing import List
from ivbase.nn.base import FCLayer

from tensornet.utils import evaluate_input, batch_node, tensor_norm, create_tensor, chain_matmul_square
from tensornet.basemodels import SimpleFeedForwardNN


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
                batch_max_parallel=4,
                output_node_position='end'):
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
        self.output_nn_size = 32
        self.output_nn_depth = 0
        
        assert(output_node_position in {'center', 'end'})
        self.output_node_position = output_node_position

        self.feature_dim = dataset[0][0].shape[-1] - 1
        
        try:
            self.output_dim = dataset[0][1].shape[-1]
        except Exception:
            self.output_dim = 0

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

        if self.output_node_position == 'center':
            self.output_core = create_tensor((bond_dim, self.output_dim, bond_dim), 
                                            requires_grad=True, opt='eye')
        elif self.output_node_position == 'end':
            self.output_core = create_tensor((bond_dim, self.output_dim), 
                                            requires_grad=True, opt='eye')

        
        output_size = self.output_nn_size if self.output_nn_depth > 0 else self.output_dim

        if self.output_node_position == 'center':
            self.output_core = create_tensor((bond_dim, output_size, bond_dim), 
                                            requires_grad=True, opt='norm')
        elif self.output_node_position == 'end':
            self.output_core = create_tensor((bond_dim, output_size), requires_grad=True, opt='norm')
            
        self.output_core = torch.nn.Parameter(self.output_core)

        self.softmax_temperature = torch.nn.Parameter(torch.Tensor([10.0]).float())

        # Initializing neural network layers for the inputs
        input_nn_kwargs = {} if input_nn_kwargs is None else input_nn_kwargs
        self.input_nn = SimpleFeedForwardNN(
                                        depth=input_nn_depth, in_size=self.feature_dim, 
                                        out_size=input_nn_out_size,
                                        activation='relu', last_activation='none', 
                                        **input_nn_kwargs)
        
        self.output_nn = SimpleFeedForwardNN(
                                        depth=self.output_nn_depth, in_size=self.output_nn_size, 
                                        out_size=self.output_dim,
                                        activation='relu', last_activation='none', 
                                        **input_nn_kwargs)

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

        # self.tensor_core.data = self.tensor_core / tensor_norm(self.tensor_core)

        #output = [self._forward(inputs[ii:ii+factor]) for ii in range(0, inputs.shape[0], factor)]
        output = [self._altforward(inputs[ii:ii+factor]) for ii in range(0, inputs.shape[0], factor)]
        output = torch.cat(output, dim=0)

        return output
        

    def _forward(self, inputs: torch.Tensor):

        inputs = self.apply_nn(inputs)
        
        #slice the inputs tensor in the input_len dimension
        input_len = inputs.size(1)
        input_list = [inputs.select(1,i) for i in range(input_len)]
        input_list = [tn.Node(vect) for vect in input_list]
        input_node_list = [tn.Node(self.tensor_core) for i in range(input_len)]

        if self.output_dim > 0:
            output_node = tn.Node(self.output_core, name = 'output') 

            if self.output_node_position == 'center':
                #add output node at the center of the input nodes
                node_list = input_node_list.copy()
                node_list.insert(input_len//2, output_node)
            elif self.output_node_position == 'end':
                node_list = input_node_list + [output_node]
        elif self.output_dim == 0:
            node_list = input_node_list

        #connect tensor cores
        for i in range(len(node_list)-1):
            node_list[i][2]^node_list[i+1][0]

        #connect the alpha and omega nodes to the first and last nodes
        tn.Node(self.alpha, name = 'alpha')[0]^node_list[0][0]

        if self.output_node_position != 'end':
            tn.Node(self.omega, name = 'omega')[0]^node_list[len(node_list)-1][2]

        output = evaluate_input(input_node_list, input_list, dtype=self.dtype).tensor

        output = self.output_nn(output)

        return output

    def apply_nn(self, inputs: torch.Tensor):
        #The slice inputs[:,:,0] has 0 for normal inputs and 1 for padding vectors.
        #We need the FC nn to preserve this.
        if self.input_nn_depth > 0:
            nned_inputs = inputs[:,:,1:]
            nned_inputs = self.input_nn(nned_inputs)

            d1,d2,d3 = nned_inputs.shape
            new_inputs = torch.zeros(d1,d2,d3+1, dtype=self.dtype).type_as(inputs)
            new_inputs[:,:,0] = inputs[:,:,0]
            new_inputs[:,:,1:] = nned_inputs
            inputs = new_inputs
            # inputs = F.layer_norm(inputs, inputs.shape[-1:])
            inputs = F.softmax(inputs*self.softmax_temperature, dim=-1)
        return inputs

    def _altforward(self, inputs: torch.Tensor):
        '''Each inputs x_i are contracted with a tensor A core first resulting in the matrix A_i.
        The matrix A_i is then normalised to prevent exponential growth of the norm of the
        intermediate states of the MPS. 
        Args:
            inputs:     A torch tensor of dimensions (batch_dim, input_len, feature_dim)
        
        Returns:        A torch tensor of dimensions (batch_dim, output_dim)
        '''
        inputs = self.apply_nn(inputs)
        batch_size, input_len, feature_dim  = inputs.shape
        core = self.tensor_core #has dimension (bond_dim, feature_dim, bond_dim)
        bond_dim = self.bond_dim
    

        #The contraction of inputs and core along the feature_dim dimension is computed. 
        #The result has dimension (batch_dim,input_len,bond_dim,bond_dim)
        matrix_stack = torch.einsum('ijk,lkm->ijlm',inputs, core) 

        #If input_len is large, matrix product states run the risk of overflowing when
        #the result of the contraction is computed. We normalise the matrices
        #to stabilize the computations.
        #We compute the norm of the matrices in the (bond_dim,bond_dim) indices
        matrix_norms = torch.zeros(batch_size, input_len, 1,1)
        matrix_norms[:,:,0,0] = torch.norm(matrix_stack,dim=(2,3))/bond_dim
        #and divide the matrix_stack by their matrix_norms
        matrix_norms = matrix_norms.expand(batch_size,input_len,bond_dim,bond_dim)
        matrix_stack = matrix_stack/matrix_norms
        
        #put the output node on the nodes list
        matrix_stack = matrix_stack.reshape(input_len, batch_size, bond_dim,bond_dim)
        if self.output_dim > 0:

            if self.output_node_position == 'center':
                left, right = matrix_stack.split(math.ceil(input_len/2),dim=0)
                left = chain_matmul_square(left)
                right = chain_matmul_square(right)
                prod = torch.einsum('i,ijk,klm->jlm',self.alpha,left,self.output_core)
                prod = torch.einsum('jlm,mjo,o->jl')
                return prod
            elif self.output_node_position == 'end':
                prod = chain_matmul_square(matrix_stack)
                prod = torch.einsum('j,ijk,kl->il', self.alpha, prod, self.output_core)
                return prod
                
        elif self.output_dim == 0:
            prod = chain_matmul_square(matrix_stack)
            prod = torch.einsum('i,ijk,k->j', self.alpha, prod, self.omega)
            return prod


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
        self.output_dim = dataset[0][1].shape[-1]

        # Initializing neural network layers for the outputs
        output_nn_kwargs = {} if output_nn_kwargs is None else output_nn_kwargs
        nn_size = self.output_dim * output_n_umps
        self.output_nn = SimpleFeedForwardNN(
                                depth=output_depth, 
                                in_size=nn_size, 
                                out_size=self.output_dim,
                                activation='relu', last_activation='none', 
                                **output_nn_kwargs)
        
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
        output = self.output_nn(umps_results)
       
        return output

    def to(self, dtype):
        super().to(dtype)
        for net in self.umps:
            net.to(dtype)
        
        return self

if __name__=='__main__':
    from tensornet.regressor import _collate_with_padding
    from tensornet import dataset
    
     