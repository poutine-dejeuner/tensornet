import numpy, torch
import tensornetwork as tn

from tensornetwork import FiniteMPS

from typing import List

tn.set_default_backend("pytorch")
torch.set_default_tensor_type(torch.DoubleTensor)

class UMPS(torch.nn.Module):

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
        """

        super(torch.nn.Module,self).__init__()

        self.feature_dim = feature_dim
        self.output_dim = output_dim
        self.bond_dim = bond_dim

        self.tensor_core = torch.randn(bond_dim,feature_dim,bond_dim)
        self.alpha = torch.randn(bond_dim)
        self.omega = torch.randn(bond_dim)

        self.label_core = torch.randn(bond_dim,feature_dim,bond_dim)

    def forward(self, inputs: List[torch.Tensor]):
        if self.output_dim == 0:
            input_len = len(inputs)
            input_list = [tn.Node(vect) for vect in inputs]
            node_list = [tn.Node(self.tensor_core) for i in range(input_len)]

        #connect tensor cores
        for i in range(input_len-1):
            node_list[i][2]^node_list[i+1][0]

        #connect the alpha and omega nodes to the first and last nodes
        tn.Node(self.alpha)[0]^node_list[0][0]
        tn.Node(self.omega)[0]^node_list[input_len-1][2]

        return evaluate_input(node_list, input_list).tensor

    def forward_old(self, inputs: List[torch.Tensor]):
        if self.output_dim == 0:
            input_len = len(inputs)
            input_list = [tn.Node(vect) for vect in inputs]
            core_list = [tn.Node(self.tensor_core) for i in range(input_len)]

            #connect a core node with the next core node and with its input node
            for i in range(input_len-1):
                core_list[i][2]^core_list[i+1][0]
                core_list[i][1]^input_list[i][0]

            #connect the last core node with its input node
            core_list[input_len-1][1]^input_list[input_len-1][0]

            #connect the alpha and omega nodes to the first and last nodes
            tn.Node(self.alpha)[0]^core_list[0][0]
            tn.Node(self.omega)[0]^core_list[input_len-1][2]

            #contracts the network
            net = tn.reachable(core_list[0])
            contracted_node = tn.contractors.auto(net)

            return contracted_node.tensor
        elif output_dim>0:
            return

def evaluate_input(node_list, input_list):
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
    num_cores = len(node_list)
    assert len(input_list) == num_cores
    assert len(set(len(inp.shape) for inp in input_list)) == 1

    # Get batch information about our input
    input_shape = input_list[0].shape
    has_batch = len(input_shape) == 2
    assert len(input_shape) in (1, 2)
    if has_batch: 
        batch_dim = input_shape[0]
        assert all(i.shape[0] == batch_dim for i in input_list)

        # Generate copy node for dealing with batch dims
        batch_edges = batch_node(num_cores, batch_dim)
        assert len(batch_edges) == num_cores + 1

    # Go through and contract all inputs with corresponding cores
    for i, node, inp in zip(range(num_cores), node_list, input_list):
        inp_node = tn.Node(inp)
        node[1] ^ inp_node[int(has_batch)]

        # Explicitly contract batch indices together if we need that
        if has_batch:
            inp_node[0] ^ batch_edges[i]
    
    contractor = tn.contractors.auto

    return contractor(tn.reachable(node_list))            

def batch_node(num_inputs, batch_dim):
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
    # For small numbers of edges, just use a single CopyNode. Every pair of 
    # input nodes will be connected to a copy node with 3 edges thus reducing
    # the number of free edges by one each times and this process is repeated 
    # iteratively until only one free edge remains.
    num_edges = num_inputs + 1
    if num_edges < 4:
        node = tn.CopyNode(rank=num_edges, dimension=batch_dim)
        return node.get_all_edges()
    
    # Initialize list of free edges with output of trivial identity mats
    input_node = tn.Node(torch.eye(batch_dim))
    edge_list, dummy_list = zip(*[input_node.copy().get_all_edges() 
                                    for _ in range(num_edges)])
    
    # Iteratively contract dummy edges until we have less than 4
    dummy_len = len(dummy_list)
    while dummy_len > 4:
        odd = dummy_len % 2 == 1
        half_len = dummy_len // 2
    
        # Apply third order tensor to contract two dummy indices together
        temp_list = []
        for i in range(half_len):
            temp_node = tn.CopyNode(rank=3, dimension=batch_dim)
            temp_node[1] ^ dummy_list[2 * i]
            temp_node[2] ^ dummy_list[2 * i + 1]
            temp_list.append(temp_node[0])
        if odd:
            temp_list.append(dummy_list[-1])
    
        dummy_list = temp_list
        dummy_len = len(dummy_list)
    
    # Contract the last dummy indices together
    last_node = tn.CopyNode(rank=dummy_len, dimension=batch_dim)
    [last_node[i] ^ dummy_list[i] for i in range(dummy_len)]
    
    return edge_list

#TODO: torch.register_parameters?

if __name__=='__main__':

    mps = UMPS(2,2)
    x = [torch.rand(2,2) for i in range(8)]
    print(mps.forward(x))
    