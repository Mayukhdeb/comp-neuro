import torch.nn as nn
import matplotlib.pyplot as plt
from .utils.seed import seed_everything

def build_mlp_model(size_sequence: list[int], activation=None, bias = False):
    """
    Create a PyTorch model based on the given size sequence, with a custom activation function.

    Parameters:
        size_sequence (list): A list of integers where each integer represents the number of neurons in each layer.
        activation (callable): The activation function to use between the linear layers. Defaults to nn.ReLU().

    Returns:
        nn.Sequential: A PyTorch Sequential model consisting of linear layers with specified activations between them.
    """
    seed_everything(0)
    layers = []
    num_layers = len(size_sequence)
    if activation is None:
        activation = nn.Identity()
    
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

def visualize_mlp(model, save_as: str = None, fig_width=10, fig_height=8):
    """
    Visualize the model weights in matplotlib with nodes as units and lines as weights absolute values.

    Parameters:
        model (nn.Sequential): PyTorch Sequential model to visualize
        save_as (str, optional): If provided, save the visualization to this file path
        fig_width (float): Width of the figure in inches
        fig_height (float): Height of the figure in inches
    """
    linear_layers = [layer for layer in model if isinstance(layer, nn.Linear)]
    num_layers = len(linear_layers)
    num_units = [linear_layers[0].in_features] + [layer.out_features for layer in linear_layers]
    max_units = max(num_units)
    # Auto-determine figure size based on network architecture
    auto_width = max(fig_width, num_layers * 2)  # Scale width by number of layers
    auto_height = max(fig_height, max_units * 0.75)  # Scale height by maximum layer size
    
    # Create figure with calculated size
    fig, ax = plt.subplots(figsize=(auto_width, auto_height))
    ax.axis('off')
    
    # Calculate the horizontal spacing between layers
    x_spacing = 1.0
    
    # Calculate the maximum vertical space needed
    max_units = max(num_units)
    y_unit_spacing = 1.0  # Space between nodes in a layer
    
    # Set consistent boundaries for the plot
    x_min = -0.5
    x_max = (num_layers) * x_spacing + 0.5
    y_min = -max_units * y_unit_spacing / 2 - 0.5
    y_max = max_units * y_unit_spacing / 2 + 0.5
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    # Store node positions
    nodes_positions = {}
    for layer_idx in range(num_layers + 1):
        layer_units = num_units[layer_idx]
        for unit_idx in range(layer_units):
            # Center nodes vertically and space them evenly
            if layer_units > 1:
                y_pos = unit_idx * y_unit_spacing - (layer_units - 1) * y_unit_spacing / 2
            else:
                y_pos = 0
            
            x_pos = layer_idx * x_spacing
            nodes_positions[(layer_idx, unit_idx)] = (x_pos, y_pos)
            
            # Draw node
            circle = plt.Circle((x_pos, y_pos), 0.1, fill=True, color='blue', clip_on=False)
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
                        color=color, linewidth=thickness, alpha=0.6)

    ax.set_title(f"MLP Architecture: {' → '.join(map(str, num_units))}")
    
    # Maintain aspect ratio to avoid distortion
    # ax.set_aspect('equal')
    
    plt.tight_layout()
    
    if save_as:
        plt.savefig(save_as, dpi=300, bbox_inches='tight')
    
    return fig, ax