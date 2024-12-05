class LayerGenerator():
    """Base class for all layer generators"""
    def generate_layer_class(self):
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def generate_layer_instance(self, layer_name, attributes):
        raise NotImplementedError("This method should be implemented by subclasses.")
    
class LinearLayerGenerator(LayerGenerator):
    def generate_layer_class(self):
        return """class Linear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
    
    def forward(self, x):
        return self.linear(x)
"""
    
    def generate_layer_instance(self, layer_name, attributes):
        in_features = attributes['in_features']['value']
        out_features = attributes['out_features']['value']
        return f"        self.{layer_name} = Linear({in_features}, {out_features})\n"

class Conv2dLayerGenerator(LayerGenerator):
    def generate_layer_class(self):
        return """class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
    
    def forward(self, x):
        return self.conv(x)
"""

    def generate_layer_instance(self, layer_name, attributes):
        in_channels = attributes['in_channels']['value']
        out_channels = attributes['out_channels']['value']
        kernel_size = attributes['kernel_size']['value']
        return f"        self.{layer_name} = Conv2d({in_channels}, {out_channels}, {kernel_size})\n"

