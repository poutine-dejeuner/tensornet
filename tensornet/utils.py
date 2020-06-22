import torch
import tensornetwork as tn
import numpy as np
import pandas as pd
import math

from typing import Any
from tensornetwork.backends.pytorch.pytorch_backend import PyTorchBackend

Tensor = Any

def tensor_tree_node_init(shape, std=1e-8):
    diag = torch.zeros(shape[1:])
    #set ones on diagonal
    idx = list(range(shape[1]))
    idx = [idx for j in range(len(shape)-1)]
    diag[idx] = 1
    diag = diag.expand(shape)
    noise = math.sqrt(std)*torch.randn(shape)
    tensor = diag + noise
    return tensor

def create_tensor(shape, opt='eye', dtype=torch.float, **kwargs):
    if opt == 'eye':
        tensor = random_tensor(shape, **kwargs)
    elif opt == 'norm':
        tensor = torch.randn(shape, **kwargs)
        tensor = tensor / tensor_norm(tensor)
    
    return tensor.to(dtype)

def random_tensor(shape, std = 1e-8, **kwargs):
    """
    This method returns a tensor in the given shape. It is an eye matrix in dimensions (0,2) and
    expanded in the 1 dimension. Some noise is added.
    """
    noise = math.sqrt(std)*torch.randn(shape, **kwargs)
    if len(shape)==2:
         return torch.eye(*shape) + noise
    ten = torch.eye(shape[0],shape[2]).reshape((shape[0], 1, shape[2]))
    ten = ten.expand(shape)
    ten = ten + noise
    return ten

def evaluate_input(node_list, input_list, dtype=torch.float):
    """
    Contract input vectors with tensor network to get scalar output
​
    Args:
        node_list:   List of nodes with dangling edges of a tensor network
        input_list:  List of inputs for each of the nodes in node_list.
                     When processing a batch of inputs, a list of matrices 
                     with shapes (batch_dim, input_dim_i) or a single 
                     tensor with shape (num_cores, batch_dim, input_dim)
                     can be specified
​
    Returns:
        closed_list: List of tensors that encodes the closed tensor network
    """
    device = node_list[0].tensor.device
    num_cores = len(node_list)
    assert len(input_list) == num_cores
    assert len(set(len(inp.shape) for inp in input_list)) == 1

    # Get batch information about our input
    input_shape = input_list[0].shape
    has_batch = len(input_shape) == 2
    
    if has_batch: 
        batch_dim = input_shape[0]
        assert all(i.shape[0] == batch_dim for i in input_list)

        # Generate copy node for dealing with batch dims
        batch_edges, _ = batch_node(num_cores, batch_dim, dtype=dtype, device = device)
        assert len(batch_edges) == num_cores + 1
        

    # Go through and contract all inputs with corresponding cores
    for i, node, inp in zip(range(num_cores), node_list, input_list):
        inp_node = tn.Node(inp)
        node[1] ^ inp_node[int(has_batch)]

        # Explicitly contract batch indices together if we need that
        if has_batch:
            inp_node[0] ^ batch_edges[i]
    
    net = tn.reachable(node_list)
    free_edges = tn.get_all_dangling(net)

    # The tn.contractor.auto will require an ordering of free edges if there 
    # is more than one. Here we order them so the resulting tensor will have 
    # dimension (batch_dim, output_dim)
    if len(free_edges) > 1:
        batch_edge = batch_edges[-1]
        free_edges.remove(batch_edge)
        free_edges = [batch_edge] + list(free_edges)
    
    free_edges = tuple(free_edges)
    contractor = tn.contractors.auto

    return contractor(net, free_edges)

def batch_node(num_inputs, batch_dim, device, dtype=torch.float):
    """
    Return a network of small CopyNodes which emulates a large CopyNode
    ​
    This network is used for reproducing the standard batch functionality 
    available in PyTorch, and requires connecting the `num_inputs` edges
    returned by batch_node to the respective batch indices of our inputs.
    The sole remaining free edge will then give the batch index of 
    whatever contraction occurs later with the input.
    ​
    Args:
        num_inputs: The number of batch indices to contract together
        batch_dim:  The batch dimension we intend to reproduce
    ​
    Returns:
        edge_list:  List of edges of our composite CopyNode object
    """
    # A copy node has a diagonal tensor: 1 when all indices are equal and 0 elsewhere.
    # For small numbers of edges, just use a single CopyNode. Every pair of 
    # input nodes will be connected to a copy node with 3 edges thus reducing
    # the number of free edges by one each times and this process is repeated 
    # iteratively until only one free edge remains.

    dtype_mapping = {torch.float32: np.float32, torch.float64: np.float64, torch.int64: np.int64,
                    torch.int32: np.int32, torch.int16: np.int16}

    numpy_dtype = dtype_mapping[dtype]

    num_edges = num_inputs + 1
    if num_edges < 4:
        node = tn.CopyNode(rank=num_edges, dimension=batch_dim, name = 'copy', 
                                backend = PyTorchBackendDevice(device=device), dtype=numpy_dtype)
        return node.get_all_edges(), [node]
    
    # Initialize list of free edges with output of trivial identity mats
    input_node = tn.Node(torch.eye(batch_dim, device = device), name = 'input')
    edge_list, dummy_list = zip(*[input_node.copy().get_all_edges() 
                                    for _ in range(num_edges)])
    
    # Iteratively contract dummy edges until we have less than 4
    dummy_len = len(dummy_list)
    copy_node_list = []
    while dummy_len > 4:
        odd = dummy_len % 2 == 1
        half_len = dummy_len // 2
    
        # Apply third order tensor to contract two dummy indices together
        temp_list = []
        for i in range(half_len):
            node = tn.CopyNode(rank=3, dimension=batch_dim, name = 'copy', 
                                backend = PyTorchBackendDevice(device=device), dtype=numpy_dtype)
            copy_node_list.append(node)
            node[1] ^ dummy_list[2 * i]
            node[2] ^ dummy_list[2 * i + 1]
            temp_list.append(node[0])
        if odd:
            temp_list.append(dummy_list[-1])
    
        dummy_list = temp_list
        dummy_len = len(dummy_list)
    
    # Contract the last dummy indices together
    last_node = tn.CopyNode(rank=dummy_len, dimension=batch_dim, name = 'last copy',
                                backend = PyTorchBackendDevice(device=device), dtype=numpy_dtype)
    [last_node[i] ^ dummy_list[i] for i in range(dummy_len)]
    
    return edge_list, copy_node_list

def chain_matmul_square(As):
    """
    Matrix multiplication of chains of square matrices
​
    Parameters
    --------------
        As: Tensor of shape (L, ..., N, N)
​
            The list of tensors to multiply. It supports batches.
            
            - L: Lenght of the chain of matrices for the matmul
            - N: Size of the matrix (must be a square matrix) 
​
    Returns
    ------------
        As_matmul: Tensor of shape (..., N, N)
            The tensor resulting of the chain multiplication
​
    """
    As_matmul = As
    while As_matmul.shape[0] > 1:
        if As_matmul.shape[0] % 2:
            A_last = As_matmul[-1:]
        else:
            A_last = None
        
        As_matmul = torch.matmul(As_matmul[0:-1:2], As_matmul[1::2])
        if A_last is not None:
            As_matmul = torch.cat([As_matmul, A_last], dim=0)
    
    return As_matmul.squeeze(0)

def tensor_norm(tensor):
    
    # Some simple solution that seems to not diverge
    norm = torch.sqrt(torch.sum(tensor ** 2)) ** (1/tensor.ndim)

    return norm


class TorchScalerWrapper():
    def __init__(self, scaler, dtype=torch.float32):
        self.scaler = scaler
        self.dtype = dtype
    
    def partial_fit(self, X, y=None):
        self.scaler.partial_fit(X, y)
        return self

    def fit(self, X, y=None):
        self.scaler.fit(X, y)
        return self

    def transform(self, X):
        vals = self.scaler.transform(X)
        return torch.Tensor(vals).to(self.dtype)


    def is_fitted(self):
        return self.is_fitted()

    def inverse_transform(self, X):
        vals = self.scaler.inverse_transform(X)
        return torch.Tensor(vals).to(self.dtype)

    def to(self, dtype):
        self.dtype = dtype
        return self

class PyTorchBackendDevice(PyTorchBackend):

    def __init__(self, device=torch.device('cpu')):
        super().__init__()
        self.device = device

    def convert_to_tensor(self, tensor: Tensor) -> Tensor:
        result = torch.as_tensor(tensor, device=self.device)
        return result