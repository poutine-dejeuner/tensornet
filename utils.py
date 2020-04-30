import torch
import tensornetwork as tn

def random_tensor(d1,d2,d3, std = 1e-8):
    
    ten = torch.eye(d1,d3)
    ten = ten.expand(d1,d2,d3)
    noise = torch.sqrt(std)*torch.randn(d1,d2,d3)
    ten = ten + noise
    return ten

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
    has_batch = input_shape[0] > 1
    
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
    
    net = tn.reachable(node_list)
    free_edges = tn.get_all_dangling(net)

    # The tn.contractor.auto will require an ordering of free edges if there 
    # is more than one. Here we order them so the resulting tensor will have 
    # dimension (batch_dim, output_dim)
    if len(free_edges) > 1:
        batch_edge = batch_edges[-1]
        edge = free_edges.difference({batch_edge})
        free_edges = [batch_edge]+list(edge)
                
    free_edges = tuple(free_edges)
    contractor = tn.contractors.auto

    return contractor(net, free_edges)            

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
        node = tn.CopyNode(rank=num_edges, dimension=batch_dim, name = 'copy')
        return node.get_all_edges()
    
    # Initialize list of free edges with output of trivial identity mats
    input_node = tn.Node(torch.eye(batch_dim), name = 'input')
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
            temp_node = tn.CopyNode(rank=3, dimension=batch_dim, name = 'copy')
            temp_node[1] ^ dummy_list[2 * i]
            temp_node[2] ^ dummy_list[2 * i + 1]
            temp_list.append(temp_node[0])
        if odd:
            temp_list.append(dummy_list[-1])
    
        dummy_list = temp_list
        dummy_len = len(dummy_list)
    
    # Contract the last dummy indices together
    last_node = tn.CopyNode(rank=dummy_len, dimension=batch_dim, name = 'last copy')
    [last_node[i] ^ dummy_list[i] for i in range(dummy_len)]
    
    return edge_list


def tensor_norm(tensor):
    
    # Some simple solution that seems to not diverge
    norm = torch.sqrt(torch.sum(tensor ** 2)) ** (1/tensor.ndim)

    return norm

