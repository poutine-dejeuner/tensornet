import tensornetwork as tn

tn.set_default_backend("pytorch")

def _forward(self, inputs: torch.Tensor):

        inputs = self.apply_nn(inputs)
        batch_core = torch.cat((self.eye, self.tensor_core), 1)

        #slice the inputs tensor in the input_len dimension
        input_len = inputs.size(1)
        input_list = [inputs.select(1,i) for i in range(input_len)]
        input_list = [tn.Node(vect) for vect in input_list]
        input_node_list = [tn.Node(batch_core) for i in range(input_len)]

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