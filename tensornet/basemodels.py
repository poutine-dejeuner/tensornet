import torch
from torch import nn
from ivbase.nn.base import FCLayer


class LSTMPredictor(nn.Module):

    def __init__(self, dataset, 
                    lstm_depth, lstm_hidden_size, lstm_bidirectional, lstm_kwargs=None, 
                    out_nn_depth=2, out_nn_kwargs=None):
        super().__init__()

        # Basic attributes
        self.dataset = dataset
        self.lstm_depth = lstm_depth
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_bidirectional = lstm_bidirectional
        self.num_direction = lstm_bidirectional + 1
        self.lstm_kwargs = {} if lstm_kwargs is None else lstm_kwargs
        self.out_nn_depth = out_nn_depth
        self.out_nn_kwargs = {} if out_nn_kwargs is None else out_nn_kwargs

        self.feature_dim = dataset[0][0].shape[-1]
        self.output_dim = dataset[0][1].shape[-1]

        # Generate the LSTM and output NN
        self.lstm = nn.LSTM(
                    input_size=self.feature_dim, hidden_size=lstm_hidden_size, 
                    num_layers=lstm_depth, bidirectional=lstm_bidirectional, 
                    **self.lstm_kwargs)

        nn_in_size = lstm_hidden_size * self.num_direction * lstm_depth
        self.out_nn = SimpleFeedForwardNN(
                    depth=out_nn_depth,
                    in_size=nn_in_size, 
                    out_size=self.output_dim,
                    activation='relu', last_activation='none',
                    **self.out_nn_kwargs)


    def forward(self, inputs, hx=None):
        inputs = inputs.permute(1, 0, 2)
        _, (hn, cn) = self.lstm.forward(inputs, hx=hx)

        output = cn
        output = output.permute(1, 0, 2)
        output = output.reshape(output.shape[0], -1)

        output = self.out_nn(output)
        
        return output



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


