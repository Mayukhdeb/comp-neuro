import torch.nn as nn

def build_mlp_model(size_sequence, activation=nn.ReLU()):
    """
    Create a PyTorch model based on the given size sequence, with a custom activation function.

    Parameters:
        size_sequence (list): A list of integers where each integer represents the number of neurons in each layer.
        activation (callable): The activation function to use between the linear layers. Defaults to nn.ReLU().

    Returns:
        nn.Sequential: A PyTorch Sequential model consisting of linear layers with specified activations between them.
    """
    layers = []
    num_layers = len(size_sequence)
    
    # Loop through the size_sequence list to add Linear and activation layers
    for i in range(num_layers - 1):
        # Add the linear layer
        layers.append(nn.Linear(size_sequence[i], size_sequence[i+1]))
        
        # Add the activation layer after each linear layer except the last one
        if i < num_layers - 2:
            layers.append(activation)
    
    # Create a Sequential model using the layers list
    model = nn.Sequential(*layers)
    return model