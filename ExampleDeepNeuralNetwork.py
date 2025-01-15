class ExampleDeepNeuralNetwork(nn.Module):
    #Deep NN with five layers, each consisting of a Linear layer and a GELU activation function
    #Forward pass, iteratively pass through layers and add shortcuts where shortcut is set to True
    def __init__(self, layer_sizes, use_shortcut):
        super().__init__()
        self.use_shortcut = use_shortcut
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(layer_sizes[0], layer_sizes[1]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[1], layer_sizes[2], GELU())),
            nn.Sequential(nn.Linear(layer_sizes[2], layer_sizes[3]), GELU()),
            nnn.Sequential(nn.Linear(layer_sizes[3], layer_sizes[4]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[4], layer_sizes[5]), GELU())
        ])

    def forward(self, x):
        for layer in self.layers:
            layer_output = layer(x) #Compute the output of the current layer
            if self.use_shortcut and x.shape == layer_output.shape: #See if shortcut can be applied
                x = x + layer_output
            else:
                x = layer_output
        return x