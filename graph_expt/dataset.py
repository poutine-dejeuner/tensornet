import torch

from gnnfp.utils import GraphFPDataset

class MolGraphDataset(GraphFPDataset):
    def __init__(self, 
                data_path, 
                features_path, 
                cache_file_path=None, 
                smiles_column="smiles", 
                ignore_fails=True
                ):
        super().__init__(data_path = data_path,
                        features_path = features_path,
                        cache_file_path = cache_file_path,
                        smiles_column = smiles_column,
                        ignore_fails = ignore_fails)
        self.vocabulary = self.build_fragment_vocabulary()
        self.vocab_len = len(self.vocabulary)

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
        Takes a DGL graph object et returns a list of one hot vectors for each node.
        """
        features_list = []
        for node in graph.nodes_dict:
            smiles = graph.nodes_dict[node]['smiles']
            index = self.vocabulary.index(smiles)
            feature = torch.zeros(self.vocab_len)
            feature[index] = 1
            features_list.append(feature)
        
        return features_list

if __name__ == '__main__':
    import os, tensornet
    from tensornet.utils import build_fragment_vocabulary
    from gnnfp.utils import GraphFPDataset

    datapath = os.path.join(os.path.dirname(tensornet.__path__._path[0]), 'data/qm9_mini.csv')
    featurepath = os.path.join(os.path.dirname(tensornet.__path__._path[0]), 'data/qm9_mini/tree.db')
    dataset = GraphFPDataset(datapath, featurepath)
    vocab = build_fragment_vocabulary(dataset)
    dataset = MolGraphDataset(data_path=datapath, features_path=featurepath)
    print(dataset.__getitem__(32))