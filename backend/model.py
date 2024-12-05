import torch
import torch.nn as nn
from collections import deque
from layer_generator import LinearLayerGenerator, Conv2dLayerGenerator

class ModelGenerator:
    def __init__(self, node_dict, connection_map, in_degree):
        self.node_dict = node_dict
        self.connection_map = connection_map
        self.in_degree_dict = in_degree
        self.execution_order = self._determine_execution_order()
        self.layer_generators = {
            'linearNode': LinearLayerGenerator(),
            'conv2dNode': Conv2dLayerGenerator()
        }
    
    def _determine_execution_order(self):
        """Determine the order in which layers should be executed"""
        order = []
        queue = deque([node_id for node_id, degree in self.in_degree_dict.items() if degree == 0])
        
        while queue:
            current_node = queue.popleft()
            order.append(current_node)
            
            next_nodes = self.connection_map.get(current_node, [])
            for next_node in next_nodes:
                self.in_degree_dict[next_node] -= 1
                if self.in_degree_dict[next_node] == 0:
                    queue.append(next_node)
        
        return order

    def generate_model(self):
        """Generate both the code and a runnable model instance"""
        # Generate the code
        code = self._generate_code()

        '''You can use this code if you want to return the model, but it's not necessary rn because we could just run the code if it's needed'''
        # # Create a new module and load the code
        # module = types.ModuleType('generated_model')
        # module.torch = torch
        # module.nn = torch.nn
        
        # # Execute the code in the module's namespace
        # exec(code, module.__dict__)
        
        # # Create an instance of the model
        # model = module.MyModel()
        
        return {
            'code': code,
            # 'model': model
        }
    
    def _generate_code(self):
        """Generate clean PyTorch code"""
        code = "import torch.nn as nn\n\n"
        
        # Generate and include all unique layer class definitions
        unique_classes = set()
        for node_type in set(node[0] for node in self.node_dict.values()):
            print(f"Node type: {node_type}")
            if node_type in self.layer_generators:
                layer_class_code = self.layer_generators[node_type].generate_layer_class()
                if layer_class_code not in unique_classes:
                    unique_classes.add(layer_class_code)
                    print(f"Adding layer class code: {layer_class_code}")
                    code += layer_class_code + "\n"

        # Generate main model class
        code += """class MyModel(nn.Module):
    def __init__(self):
        super().__init__()\n"""
        
        # Add layer instances
        for idx, node_id in enumerate(self.execution_order, 1):
            node_type, attributes, *_ = self.node_dict[node_id]
            layer_name = f"layer{idx}"
            
            if node_type in self.layer_generators:
                code += self.layer_generators[node_type].generate_layer_instance(layer_name, attributes)

        # Add forward method
        code += "\n    def forward(self, x):\n"
        for idx, node_id in enumerate(self.execution_order, 1):
            #layer_name = f"{self.node_dict[node_id][0].lower()}{idx}"
            code += f"        x = self.layer{idx}(x)\n"
        code += "        return x\n\n"
        
        return code