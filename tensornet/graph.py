import torch
import tensornetwork as tn
import networkx as nx
import pytorch_lightning as pl
import numpy as np

from tensornet.utils import tensor_tree_node_init
from gnnfp.utils import GraphFPDataset


tn.set_default_backend("pytorch")
torch.set_default_tensor_type(torch.FloatTensor)

class tree_node(object):
    def __init__(self, data):
        self.data = data
        self.children = []

    def add_child(self, obj):
        self.children.append(obj)

    def add_children_from_list(self, child_list):
        self.children = self.children + child_list

class StaticGraphTensorNetwork(pl.LightningModule):
    
    def __init__(self, 
                dataset,
                bond_dim = 20,
                max_degree = 4,
                max_depth = 5,
                embedding_dim = 32,
                std = 1e-8,
                uniform = False
        ):
        '''
        Parameters
            ----------
            bond_dim: the dimension of the axis shared by the non input nodes in the network.
            max_degree: the nodes will have max_degree + 1 indices, one for the input and the others to connect 
            with the non input nodes.
            max_depth: this is the number of levels in the graph. if max_depth=1, there is 1 node, if max_depth=2
            there are 1+max_degree nodes, if max_depth=3 there are 1 + max_degree + max_degree(max_degree-1).
            output_dim: the dimension of the output.
            embedding_dim: the dimension of the embedding of the inputs.
        '''
        super().__init__()
        self.dataset = dataset
        self.bond_dim = bond_dim
        self.max_degree = max_degree
        self.max_depth = max_depth
        self.input_dim = embedding_dim
        self.output_dim = len(dataset.__getitem__(0)['labels'])
        self.uniform = uniform
        
        self.tensor_list, self.network, self.edges = all_random_diag_init(
            self.input_dim, bond_dim, max_degree, max_depth, self.output_dim, std, uniform = self.uniform)

        self.embedding = torch.nn.Embedding(num_embeddings = dataset.vocab_len,
                                            embedding_dim = embedding_dim)

    def forward(self, mol_graph, features):
        """
        makes a copy of self.network, connects the graphs feature vectors embedding mol_graph in the network 
        and puts padding vectors at all other inputs. Contracts the network.
        """
        network = self.network
        self.connect_nodes()
        features = self.embedding(features)
        network = connect_inputs(network, mol_graph, features, self.input_dim)
        while len(network)>1:
            for node in network:
                if len(node.tensor.shape)==1:
                    edge = node.edges[0]
                    network.remove(edge.node1)
                    network.remove(edge.node2)
                    new_node = tn.contract(edge)
                    network.append(new_node)

        return network[0].tensor

    def connect_nodes(self):
        """
        Connects the core nodes of the network. This operation needs to be done at each call of 
        forward since the contraction step at the end of forward disconnects the initial nodes.
        """
        if isinstance(self.network, list):
            for edge in self.edges:
                node0, node1 = edge
                edge0 = self.network[node0].get_all_dangling()[1]
                edge1 = self.network[node1].get_all_dangling()[1]
                edge0 ^ edge1
        
        elif isinstance(self.network, tree_node):
            parents = [self.network]
            while parents != []:
                node = parents.pop()
                for child in node.children:
                    edge1 = node.data.get_all_dangling()[1]
                    edge2 = child.data.get_all_dangling()[1]
                    edge1 ^ edge2
                    parents = [child] + parents

def connect_inputs(network, mol_graph, features, input_dim):
    """
    Connects the molecular graph features (inputs) to the network by embedding the molecular graph into the 
    tensor network graph. The center of the molecular graph is put at the center of the network.
    The nodes of the mol_graph are molecular fragments even if they are called atoms in the comments.
    It works like this: Start with the center of the mol_graph and the center of the network. Those 2 nodes
    are identified in the graph_net_dict. Compute the children of the nodes in their respective graphs. 
    The network node is attached to the input corresponding to the feature of the central node in mol_graph.
    From the mol_child_dict compute the next mol_child_dict. For every child in mol_child_dict, use mol_net_dict
    to find the node in the network corresponding to its parent. pick a node in the network that will represent
    the child node in mol_child_dict and pair them in the mol_net_dict. compute the new net_child_dict and 
    connect the corresponding feature vector.
    
    Args: mol_graph: a dgl graph of molecular fragments.

    returns: network: a list of connected tensornetwork nodes.
    
    """
    mol_graph = mol_graph.to_networkx()
    # first connect the input of the center of the molecule to the center of the network
    center = nx.algorithms.distance_measures.center(mol_graph)[0]
    mol_child_dict = {center: list(nx.neighbors(mol_graph, center))}

    if isinstance(network, tree_node):
        input_nodes = [tn.Node(features[center])]
        node = network.data
        node[0] ^ input_nodes[0][0]
        mol_net_dict = {center: network}  
        while mol_child_dict != dict():
            #connect features of atoms in to the child nodes. parents have been connected in previous step
            for parent in mol_child_dict:
                for i, child in enumerate(mol_child_dict[parent]):
                    input_node = tn.Node(features[child])
                    input_nodes.append(input_node)
                    node = mol_net_dict[parent].children[i]
                    node.data[0] ^ input_node[0]
                    assert(input_node.get_all_dangling()==[])
                    #update mol_child_dict with child as new parent
                    mol_child_dict[child] = get_graph_children(mol_graph, mol_child_dict)
                mol_child_dict.popitem(parent)

    elif isinstance(network, list):
        input_node = tn.Node(features[center])
        input_nodes = [input_node]
        net_child_dict = {network[-1]: tn.get_neighbors(network[-1])}
        network[-1]['input'] ^ input_node[0]
        #this dictionary pairs atoms in mol_graph with nodes of network
        graph_net_dict = {center:network[-1]}
        
        while mol_child_dict != dict():
            #we need to get the next net_child_dict before connecting nodes otherwise the neighborhood 
            #will contain input nodes too
            next_net_child_dict = get_graph_children(network, net_child_dict)
            #connect features of atoms in atom_list to network nodes
            for parent in mol_child_dict:
                net_parent = graph_net_dict[parent]
                net_children = net_child_dict[net_parent]
                for child in mol_child_dict[parent]:
                    node = None
                    for node in net_children:
                        if node['input'].is_dangling() == True:
                            break
                    graph_net_dict[child] = node
                    input_node = tn.Node(features[child])
                    node['input'] ^ input_node[0]
                    input_nodes.append(input_node)
        
            #get children of atoms in atom_list. A parent list is needed to remove them from 
            #the neighbors to get the next iterations child list.
            mol_child_dict = get_graph_children(mol_graph, mol_child_dict)
            net_child_dict = next_net_child_dict

    dangling_edges = tn.get_all_dangling(network)
    for edge in dangling_edges:
        if edge.name == 'output':
            dangling_edges.remove(edge)
    padding_vect = torch.zeros(input_dim)
    padding_vect[0] = 1
    
    for edge in dangling_edges:
        padding_node = tn.Node(padding_vect)
        edge ^ padding_node[0]
        input_nodes.append(padding_node)
    network = network + input_nodes
    return network

def all_random_diag_init(input_dim, bond_dim, max_degree, max_depth, output_dim, std, uniform):
    """
    Initialises the tensors of the network and connects them, leaving the input edges free. 
    Initialisation of tensors is done with utils.tensor_tree_node_init.
    The first index of the nodes is the input, the other are the bonds.
    """
    from copy import deepcopy
    #Create the graph G for the tensor network
    node = nx.Graph()
    node.add_node(1)
    branch = nx.generators.classic.balanced_tree(max_degree - 1, max_depth -1)
    size = len(branch)
    G = nx.Graph()
    for i in range(max_degree):
        G = nx.disjoint_union(G, branch)

    G = nx.disjoint_union(G, node)
    G.add_edges_from([(0,4*size), (size, 4*size), (2*size,4*size), (3*size,4*size)])
    edges = list(G.edges())
    assert(nx.algorithms.components.is_connected(G)==True)
    tensor_list = torch.nn.ParameterList()

    if uniform == False:
        #init core tensors and create Tensornetworks Nodes
        for node in G.nodes():
            degree = G.degree(node)
            shape = (input_dim,) + tuple(bond_dim for i in range(degree))
            tensor = torch.nn.Parameter(tensor_tree_node_init(shape, std=std))
            tensor_list.append(tensor)
        axis_names = [ ['input']+['bond'+str(i) for i in range(len(tensor.shape) - 1)] for tensor in tensor_list ]
        if output_dim>1:
            shape = (input_dim,) + tuple(bond_dim for i in range(degree)) + (output_dim,)
            tensor_list[-1] = torch.nn.Parameter(tensor_tree_node_init(shape, std=std))
            axis_names[-1] = ['input']+['bond'+str(i) for i in range(max_degree)] + ['output']
        
        network_nodes = [tn.Node(tensor, axis_names=name) for tensor, name in zip(tensor_list, axis_names)]
        network_nodes[-1].name = 'center'
    
    elif uniform == True:
        #init the leaf tensor
        shape = (input_dim, bond_dim)
        tensor = torch.nn.Parameter(tensor_tree_node_init(shape, std=std))
        tensor_list.append(tensor)
        leaf = tn.Node(tensor, name = 'leaf', axis_names = ['input','bond'])
        
        #init the regular nodes tensors
        shape = (input_dim,) + tuple(bond_dim for i in range(max_degree))
        reg_node_list = []
        for i in range(max_depth - 2):
            tensor = torch.nn.Parameter(tensor_tree_node_init(shape, std=std))
            tensor_list.append(tensor)
            axis_names = ['input'] + ['bond'+str(i) for i in range(max_degree)]
            node = tn.Node(tensor, name = 'node', axis_names = axis_names)
            reg_node_list.append(node)
        
        tree = leaf
        for node in reg_node_list:
            tree = tree_node(node)
            tree.add_children_from_list( [deepcopy(tree) for i in range(max_degree -1)] )

        #init the root tensor
        shape = (input_dim,) + tuple(bond_dim for i in range(max_degree))
        axis_names = ['input'] + ['bond'+str(i) for i in range(max_degree)]
        if output_dim > 1:
            shape = shape + (output_dim,)
            axis_names = axis_names + ['output']
        tensor = torch.nn.Parameter(tensor_tree_node_init(shape, std=std))
        tensor_list.append(tensor)
        root_node = tn.Node(tensor, name = 'root', axis_names = axis_names)

        tree = tree_node(root_node)
        tree.add_children_from_list([deepcopy(tree) for i in range(max_degree)])

        #TODO: make the tree dict the data used by the rest of the code for computations
        
        return tensor_list, tree, tree

                        
    return tensor_list, network_nodes, edges

def get_graph_children(graph, child_dict):
    """
    This takes a tree graph and a dict of parent:children and returns a dict of parent:children with the parents
    the children of the first dict
    """
    if isinstance(graph, list):
        new_child_dict = dict()
        for parent in child_dict:
            for child in child_dict[parent]:
                neighbors = list( tn.get_neighbors(child) )
                neighbors.remove(parent)
                new_child_dict[child] = neighbors
    elif isinstance(graph, nx.classes.graph.Graph):
        new_child_dict = dict()
        for parent in child_dict:
            for child in child_dict[parent]:
                neighbors = list( nx.neighbors(graph, child) )
                neighbors.remove(parent)
                new_child_dict[child] = neighbors
    
    return new_child_dict

class GraphTensorNetwork(pl.LightningModule):

    def __init__(self, 
                dataset,
                bond_dim,
                max_degree = None,
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

        self.vocabulary = dataset.vocabulary
        self.input_dim = len(self.vocabulary)
        self.tensor_cores = torch.nn.ParameterList()
        """
        The order of the tensor cores are equal to the degree of the node in the molecular graph + 1, with
        the +1 accounting for the feature input of the node. A node with no neighbors will have order 1
        (ie its a vector).
        """
        for i in range(0,max_degree + 1):
            shape = [self.input_dim] + [bond_dim for j in range(i)]
            shape = tuple(shape)
            core = tensor_tree_node_init(shape)
            core = torch.nn.Parameter(core)
            self.tensor_cores.append(core)

    def get_max_degree(self):
        max_degree_list = torch.zeros(self.dataset.__len__())
        for i in range(self.dataset.__len__()):
            edges = self.dataset.__getitem__(i)['edges']
            degrees = torch.zeros(edges.shape[0])
            for i in range(edges.shape[0]):
                degrees[edges[i][0]] += 1
                degrees[edges[i][1]] += 1
            degrees = degrees/2
            max_degree = torch.max(degrees)
            max_degree_list[i] = max_degree
        return torch.max(max_degree_list)

    def forward(self, element_data):
        edge_data = element_data['edges']
        features = element_data['features']
        nodes, atom_features = self.connect_net(edge_data, features)
        return tn.contractors.auto(nodes+atom_features)
        

    def connect_net(self, edge_data, features):
        """
        Connects the nodes of the network. Returns the connected core nodes.

        Args:
            edge_data:  adjacency matrix or 2xN tensor listing the edges of the graph.
            features:   imput vectors of the network.

        returns:
            nodes, atom_features: nodes is a list of nodes objects of TensorNetworks library that 
            have  been connected following the edge_data input.
        """
        nodes = []
        if 0 not in edge_data.shape:  
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
        elif 0 in edge_data.shape:
            deg = 0
            nodes = [tn.Node(self.tensor_cores[deg])]
            #connect nodes with inputs
            atom_features = [tn.Node(features.select(dim=0,index=i)) for i in range(features.shape[0])]
            for i, inp in enumerate(atom_features):
                input_edge = inp[0]
                node_edge = nodes[i][0]
                input_edge^node_edge

        return nodes, atom_features

class MolGraphDataset(GraphFPDataset):
    def __init__(self, 
                data_path, 
                features_path,
                num_labels=0, 
                cache_file_path=None, 
                smiles_column="smiles", 
                ignore_fails=True,
                scaler = None,
                feature_type = 'index'
                ):
        super().__init__(data_path = data_path,
                        features_path = features_path,
                        cache_file_path = cache_file_path,
                        smiles_column = smiles_column,
                        ignore_fails = ignore_fails)

        self.vocabulary = self.build_fragment_vocabulary()
        self.vocab_len = len(self.vocabulary)
        self.batch_max_parallel = 1
        self.num_labels = num_labels
        self.feature_type = feature_type

        if num_labels != -1:
            self.labels = self.labels[:,0:num_labels]
            
        self.scaler = None
        if scaler is not None:
            values = self.labels
            self.scaler = scaler.fit(values)

    def build_fragment_vocabulary(self):
        fragments = {}
        for i in range(self.__len__()):
            smile, graph, values, masks = super().__getitem__(i)
            for idx in range(len(graph.nodes_dict)):
                frag_smiles = graph.nodes_dict[idx]['smiles']
                try:
                    fragments[frag_smiles] += [smile]
                except KeyError:
                    fragments[frag_smiles] = [smile]
        return list(fragments)


    def __getitem__(self, item):
        smiles, graph, labels, mask = super().__getitem__(item)
        features = self.featurise(graph)
        edges = torch.stack(graph.edges()).T

        return {'smiles':smiles, 'graph':graph, 'edges':edges, 'features':features, 'labels':labels, 'mask':mask}

    def featurise(self, graph):
        """
        Takes a DGL graph object and returns a list of indices or one-hots for self.vocabulary for each node.
        """
        if self.feature_type == 'index':
            feature_indices = torch.LongTensor(len(graph.nodes_dict))
            for i, node in enumerate(graph.nodes_dict):
                smiles = graph.nodes_dict[node]['smiles']
                index = self.vocabulary.index(smiles)
                feature_indices[i] = index
            return feature_indices
        elif self.feature_type == 'one-hot':
            features_list = torch.zeros(len(graph.nodes_dict), self.vocab_len)
            for node in graph.nodes_dict:
                smiles = graph.nodes_dict[node]['smiles']
                index = self.vocabulary.index(smiles)
                feature = torch.zeros(self.vocab_len)
                feature[index] = 1
                features_list[node] = feature
            return features_list

if False:
    import os, tensornet
    import torch
    from gnnfp.utils import GraphFPDataset

    '''
    tensor norm test
    '''
    data_path = os.path.join(os.path.dirname(tensornet.__path__._path[0]), 'data/qm9_80.csv')
    features_path = os.path.join(os.path.dirname(tensornet.__path__._path[0]), 'data/qm9_80/tree.db')
    dataset = MolGraphDataset(data_path, features_path, num_labels=0)
    tensornet = StaticGraphTensorNetwork(dataset, bond_dim = 10)
    graph = dataset.__getitem__(32)['graph']
    features = dataset.__getitem__(32)['features']
    prod = tensornet(graph,features)
    print(prod)

if __name__ == '__main__':
    import os, tensornet
    import torch
    from gnnfp.utils import GraphFPDataset
    
    data_path = os.path.join(os.path.dirname(tensornet.__path__._path[0]), 'data/qm9_80.csv')
    features_path = os.path.join(os.path.dirname(tensornet.__path__._path[0]), 'data/qm9_80/tree.db')
    dataset = MolGraphDataset(data_path, features_path, num_labels=0)
    
    
    print('----------------------------')
    print('dataset test: list data sample')
    print(dataset.__getitem__(16))

    '''
    print('dynamic graph tensor network test')
    moltennet = GraphTensorNetwork(
        dataset=dataset,
        max_degree=4,
        bond_dim=20,
        edge_structure='list'
        )

    for i in range(4,16):
        data = dataset.__getitem__(i)
        result = moltennet(data)
        print(result.tensor)
    '''
    print('----------------------------')
    print('static graph tensor net test')
    tensornet = StaticGraphTensorNetwork(dataset, bond_dim = 10)
    data = dataset.__getitem__(32)
    print(tensornet(data['graph'],data['features']))
    data = dataset.__getitem__(33)
    print(tensornet(data['graph'],data['features']))

    print('----------------------------')
    print('uniform static graph tensor net test')
    tensornet = StaticGraphTensorNetwork(dataset, bond_dim = 10, uniform=True)
    graph = dataset.__getitem__(16)['graph']
    features = dataset.__getitem__(16)['features']
    tensornet = StaticGraphTensorNetwork(dataset, bond_dim = 10)
    print(tensornet(graph,features))

    print('----------------------------')
    print('static graph tensor net multiple labels test')
    dataset = MolGraphDataset(data_path, features_path, num_labels=-1)
    print('num labels=' + str(len(dataset.__getitem__(16)['labels'])))
    graph = dataset.__getitem__(16)['graph']
    features = dataset.__getitem__(16)['features']
    tensornet = StaticGraphTensorNetwork(dataset, bond_dim = 10)
    print(tensornet(graph,features))

    print('----------------------------')
    print('get_graph_children test')

    print('dgl graph:')
    node = nx.Graph()
    node.add_node(1)
    branch = nx.generators.classic.balanced_tree(3,2)
    size = len(branch)
    G = nx.disjoint_union(branch, branch)
    G = nx.disjoint_union(G, branch)
    G = nx.disjoint_union(G, branch)
    G.add_edges_from([(0,4*size), (size, 4*size), (2*size,4*size), (3*size,4*size)])
    center = 52
    child = nx.neighbors(G,center)
    child_dict = {center:child}
    new_child_dict = get_graph_children(G,child_dict)
    print(new_child_dict)
    assert(new_child_dict=={0: [1, 2, 3], 13: [14, 15, 16], 26: [27, 28, 29], 39: [40, 41, 42]})

    print('tensornetwork graph:')
    tensornet = StaticGraphTensorNetwork(dataset, bond_dim = 2)
    tensornet.connect_nodes()
    net = tensornet.network
    center = net[-1]
    assert(center.name=='center')
    child_dict = {net[-1]: tn.get_neighbors(net[-1])}
    child = get_graph_children(net, child_dict)
    child_list = [len(ch) for ch in child.values()]
    print(child_list)
    assert(child_list == [3,3,3,3])

import opt_einsum as oe

class MyOptimizer(oe.paths.PathOptimizer):

    def __call__(self, inputs, output, size_dict, memory_limit=None):
        '''
        Parameters
            ----------
            inputs : list[set[str]]
                The indices of each input array.
            outputs : set[str]
                The output indices
            size_dict : dict[str, int]
                The size of each index
            memory_limit : int, optional
                If given, the maximum allowed memory.
        '''
        '''
        edge = inputs[0].pop()
        node = edge.node1
        node_list = list(tn.reachable(node))
        size_list = [node.tensor.shape for node in node_list]
        min_length = min([len(size) for size in size_list])
        short_size_nodes = [node for node in node_list if len(node.tensor.shape)==min_length]
        #the nodes in short_size_nodes will be vectors so the first edge is always the one we contract
        #(there are no other edges)
        edge_list = [node.get_all_edges()[0] for node in short_size_nodes]
        edge_list = [(node_list.index(edge.node1), node_list.index(edge.node2)) for edge in edge_list]
        '''
        edge = inputs[0].pop()
        node = edge.node1
        node_list = list(tn.reachable(node))
        for node in node_list:
            if len(node.tensor.shape)==1:
                break
        edge = node.get_all_edges()[0]
        edge_list = (node_list.index(edge.node1),node_list.index(edge.node2))
        return [edge_list]
