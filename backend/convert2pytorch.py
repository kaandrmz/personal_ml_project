from collections import deque, defaultdict
from model import ModelGenerator

class GraphData:
    def __init__(self, graph_data):
        self.graph_data = graph_data


    def create_port_node_map(self, nodes):
        node_dict = defaultdict(list)
        port_node_map = defaultdict(list)
        for node in nodes:
            node_id = node['id']
            node_type = node['type']
            input_id = node['inputs']['input']['id'] 
            output_id = node['outputs']['output']['id'] # not sure if these two are actually needed
            attributes = {key: value for key, value in node['inputs'].items() if key != 'input'}

            node_dict[node_id] = [node_type, attributes, input_id, output_id]
            # add attributes such as in_features etc.?
            

            port_node_map[input_id] = node_id
            port_node_map[output_id] = node_id
        return node_dict, port_node_map # {node_id: [type,attributes,input_id,output_id]} ; {port_id: node_id}

        
    def create_connection_map(self, connections, port_node_map): # dene bunu
        connection_map = defaultdict(list)
        for connection in connections:
            from_id = connection['from']
            to_id = connection['to']

            from_node = port_node_map.get(from_id)
            to_node = port_node_map.get(to_id)
            if from_node and to_node:
                if from_node not in connection_map:   
                    connection_map[from_node] = []  # Initialize a list for multiple outputs
                connection_map[from_node].append(to_node)
            else:
                raise ValueError(f"The node {from_node} or {to_node} is missing an input or output port!")
        return connection_map # looks like this: {node1: [node2, node3], node2: [node3], ...} (id of the nodes)
    
    
    def initialize_in_degree(self,node_dict, connection_map):
        # Initialize in-degree counter for each node based on connections
        in_degree = {node_id: 0 for node_id in node_dict}
        for to_nodes in connection_map.values():
            for to_node in to_nodes:
                in_degree[to_node] += 1
        return in_degree # {node_id: number of incoming edges}
            
   
    def build_network(self):
        nodes = self.graph_data['graph']['nodes']
        connections = self.graph_data['graph']['connections']
        node_dict, port_node_map = self.create_port_node_map(nodes)
        connection_map = self.create_connection_map(connections, port_node_map)
        in_degree = self.initialize_in_degree(node_dict, connection_map)
        # model = DynamicModel(node_dict, connection_map, in_degree)
        generator = ModelGenerator(node_dict, connection_map, in_degree)
        return generator.generate_model()
        #return self.order_nodes(node_dict, connection_map)

    # def order_nodes(self, node_dict, connection_map):
        #     '''Topological sort directed acyclic graph- Kahn's algorithm'''
        #     # Create in-degree counter and adjacency list
        #     in_degree = defaultdict(int)
        #     adjacency_list = defaultdict(list)

        #     # Initialize in-degree and adjacency list
        #     for from_node, to_nodes in connection_map.items():
        #         for to_node in to_nodes:
        #             in_degree[to_node] += 1
        #             adjacency_list[from_node].append(to_node)

        #     # Identify nodes with zero in-degree
        #     zero_in_degree_queue = deque()
        #     for node_id in node_dict.keys():
        #         if in_degree[node_id] == 0:
        #             zero_in_degree_queue.append(node_id)
            
        #     # Perform topological sorting
        #     order = []
        #     while zero_in_degree_queue:
        #         current_node = zero_in_degree_queue.popleft()
        #         order.append(current_node)
                
        #         # Decrease in-degree of connected nodes
        #         for neighbor in adjacency_list[current_node]:
        #             in_degree[neighbor] -= 1
        #             if in_degree[neighbor] == 0:
        #                 zero_in_degree_queue.append(neighbor)

        #     # Check for cycles
        #     if len(order) != len(node_dict):
        #         raise ValueError("Graph has a cycle and cannot be sorted topologically.") # In future accept also cyclic graphs? -> i.e for RNN's
            
        #     return order
