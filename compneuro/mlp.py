import torch.nn as nn
import matplotlib.pyplot as plt

def build_mlp_model(size_sequence: list[int], activation=nn.ReLU(), bias = False):
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
        layers.append(nn.Linear(size_sequence[i], size_sequence[i+1], bias=bias))
        
        # Add the activation layer after each linear layer except the last one
        if i < num_layers - 2:
            layers.append(activation)
    
    # Create a Sequential model using the layers list
    model = nn.Sequential(*layers)
    return model

def visualize_mlp(model, save_as: str = None, fig_width=10, fig_height=5):
    """
    Visualize the model weights in matplotlib with nodes as units and lines as weights absolute values.

    Parameters:
        model (nn.Sequential): PyTorch Sequential model to visualize
        save_as (str, optional): If provided, save the visualization to this file path
    """
    figsize = (fig_width, fig_height)
    linear_layers = [layer for layer in model if isinstance(layer, nn.Linear)]
    num_layers = len(linear_layers)
    num_units = [linear_layers[0].in_features] + [layer.out_features for layer in linear_layers]
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('off')
    ax.set_aspect('equal')

    # Define padding
    x_pad = 0.5  # Space on the x-axis
    y_pad = 0.2  # Space on the y-axis for centering nodes properly

    ax.set_xlim(-x_pad, num_layers + x_pad)
    ax.set_ylim(-y_pad - max(num_units) / 2, y_pad + max(num_units) / 2)

    # Store node positions
    nodes_positions = {}
    for layer_idx in range(num_layers + 1):
        layer_units = num_units[layer_idx]
        for unit_idx in range(layer_units):
            # Center y-positions dynamically
            y_pos = unit_idx - (layer_units - 1) / 2
            nodes_positions[(layer_idx, unit_idx)] = (layer_idx, y_pos)
            
            # Draw node
            circle = plt.Circle((layer_idx, y_pos), 0.1, fill=True, color='blue', clip_on=True)
            ax.add_patch(circle)

    # Draw connections between nodes
    for layer_idx, layer in enumerate(linear_layers):
        weights = layer.weight.data.cpu().numpy()
        max_weight = max(abs(weights.min()), abs(weights.max())) if weights.size > 0 else 1
        
        for in_idx in range(layer.in_features):
            for out_idx in range(layer.out_features):
                start = nodes_positions[(layer_idx, in_idx)]
                end = nodes_positions[(layer_idx + 1, out_idx)]

                weight = weights[out_idx, in_idx]
                thickness = 0.5 * abs(weight) / max_weight
                
                color = 'red' if weight < 0 else 'green'
                
                ax.plot([start[0], end[0]], [start[1], end[1]], 
                        color=color, linewidth=thickness, alpha=0.6, clip_on=True)

    ax.set_title(f"MLP Architecture: {' → '.join(map(str, num_units))}")

    if save_as:
        plt.savefig(save_as, dpi=300, bbox_inches='tight')

    plt.tight_layout()
    plt.show()

    return fig, ax