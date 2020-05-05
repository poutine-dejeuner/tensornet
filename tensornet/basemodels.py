import torch
from torch import nn
from ivbase.nn.base import FCLayer


class LSTMPredictor(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.lstm = nn.LSTM(*args, **kwargs)
        


    def forward(self, inputs, hx=None):
        inputs = inputs.permute(1, 0, 2)
        hn, cn = super().forward(inputs, hx=hx)
        
        # INCOMPLETE!!!!!!!!!!!!!!!!!!!!!!!!
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores



class SimpleFeedForwardNN(nn.Module):

    def __init__(self, depth, in_size, out_size, activation='relu', last_activation='none', **kwargs):

        super().__init__()
        
        # Hyper-parameters
        self.in_size = in_size
        self.out_size = out_size
        self.depth = depth
        self.activation = activation
        self.last_activation = last_activation
        self.kwargs = kwargs

        # Initializing neural network layers for the inputs
        self.fc_layers = nn.ModuleList()
        if depth == 0:
            pass
        elif depth == 1:
            self.fc_layers.append(FCLayer(in_size=in_size, out_size=out_size, 
                                        bias=False, activation=last_activation, **kwargs))
        elif depth >= 2:
            self.fc_layers.append(FCLayer(in_size=in_size, out_size=out_size, 
                                                bias=False, activation=activation, **kwargs))
            fc_input_layers_ext = [FCLayer(in_size=out_size, out_size=out_size, 
                                bias=False, **kwargs) for ii in range(depth - 1)]
            self.fc_layers.extend(fc_input_layers_ext)

            self.fc_layers.append(FCLayer(in_size=out_size, out_size=out_size, 
                                                bias=False, activation=last_activation, **kwargs))
        else:
            raise ValueError('`depth` must be a positive integer')
    

    def forward(self, inputs):
        for fc_layer in self.fc_layers:
            inputs = fc_layer(inputs)
        return inputs
