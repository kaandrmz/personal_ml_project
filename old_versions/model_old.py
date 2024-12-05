class LinearNode(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias)
    
    def forward(self, x):
        return self.linear(x)

def create_layer(node_type, attributes):
    if node_type == 'linearNode':
        return LinearNode(
            in_features=attributes['in_features']['value'],
            out_features=attributes['out_features']['value'],
            bias=attributes['bias']['value']
        )
    # Add more layer types here as needed
    raise ValueError(f"Unknown node type: {node_type}")

class DynamicModel(nn.Module):
    def __init__(self, node_dict, connection_map, in_degree):
        super().__init__()
        self.in_degree_dict = in_degree
        self.node_dict = node_dict
        self.connection_map = connection_map
        
        # Create and register all layers
        self.layers = nn.ModuleDict()
        self._build_layers()
        
        # Store the execution order
        self.execution_order = self._determine_execution_order()
    
    def _build_layers(self):
        """Create all layer instances and register them in the ModuleDict"""
        for node_id, (node_type, attributes, *_) in self.node_dict.items():
            self.layers[str(node_id)] = create_layer(node_type, attributes)
    
    def _determine_execution_order(self):
        """Determine the order in which layers should be executed"""
        order = []
        queue = deque([node_id for node_id, degree in self.in_degree_dict.items() if degree == 0])
        
        while queue:
            current_node = queue.popleft()
            order.append(current_node)
            
            # Get next nodes from connection map
            next_nodes = self.connection_map.get(current_node, [])
            for next_node in next_nodes:
                self.in_degree_dict[next_node] -= 1
                if self.in_degree_dict[next_node] == 0:
                    queue.append(next_node)
        
        return order

    def forward(self, x):
        # Store intermediate outputs
        outputs = {}
        
        # Process each layer in order
        for node_id in self.execution_order:
            if node_id in self.in_degree_dict and self.in_degree_dict[node_id] == 0:
                # This is an input node
                current_input = x
            else:
                # Get input from previous layer(s)
                prev_node = self._get_previous_node(node_id)
                current_input = outputs[prev_node]
            
            # Process current layer
            outputs[node_id] = self.layers[str(node_id)](current_input)
        
        # Return the output of the last layer
        return outputs[self.execution_order[-1]]
    
    def _get_previous_node(self, node_id):
        """Helper method to find the previous node in the connection map"""
        for node, connections in self.connection_map.items():
            if node_id in connections:
                return node
        return None
        