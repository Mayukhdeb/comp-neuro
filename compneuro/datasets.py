import torch
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button

def get_sine_wave_data(num_points: int, noise: float = 0.0, test_data_fraction: float = 0.2):
    x = torch.linspace(0, 2 * 3.14159, num_points).unsqueeze(1)  # Add batch dimension
    y = (torch.sin(x) + torch.randn_like(x) * noise)  # Add batch dimension
    
    x_train, x_test, y_train, y_test = train_test_split(x.numpy(), y.numpy(), test_size=test_data_fraction, random_state=0)
    
    return {
        "train": {"x": torch.tensor(x_train), "y": torch.tensor(y_train)},
        "test": {"x": torch.tensor(x_test), "y": torch.tensor(y_test)},
    }

def get_line_data(num_points: int, noise: float = 0.0, test_data_fraction: float = 0.2):
    x = torch.linspace(0, 10, num_points).unsqueeze(1)  # Add batch dimension
    y = (2 * x + 1 + torch.randn_like(x) * noise)  # Add batch dimension
    y = y /y.max()
    y = y - 0.5
    
    x_train, x_test, y_train, y_test = train_test_split(x.numpy(), y.numpy(), test_size=test_data_fraction, random_state=0)
    
    return {
        "train": {"x": torch.tensor(x_train), "y": torch.tensor(y_train)},
        "test": {"x": torch.tensor(x_test), "y": torch.tensor(y_test)},
    }

def get_line_through_zero_data(num_points: int, noise: float = 0.0, test_data_fraction: float = 0.2):
    x = torch.linspace(0, 10, num_points).unsqueeze(1)  # Add batch dimension
    y = (2 * x + 1 + torch.randn_like(x) * noise)  # Add batch dimension
    y = y /y.max()
    
    x_train, x_test, y_train, y_test = train_test_split(x.numpy(), y.numpy(), test_size=test_data_fraction, random_state=0)
    
    return {
        "train": {"x": torch.tensor(x_train), "y": torch.tensor(y_train)},
        "test": {"x": torch.tensor(x_test), "y": torch.tensor(y_test)},
    }

def get_zigzag_line_data(num_points: int, noise: float = 0.0, test_data_fraction: float = 0.2):
    x = torch.linspace(-1, 1, num_points).unsqueeze(1)  # Add batch dimension
    y = (torch.abs((x * 4 % 4) - 2) * 5 + torch.randn_like(x) * noise) # Add batch dimension
    
    x = x - x.min()
    y = y - y.min()

    x = x/x.max()
    y = y/y.max()
    x = x - 0.5
    y = y - 0.5

    x_train, x_test, y_train, y_test = train_test_split(x.numpy(), y.numpy(), test_size=test_data_fraction, random_state=0)
    
    return {
        "train": {"x": torch.tensor(x_train), "y": torch.tensor(y_train)},
        "test": {"x": torch.tensor(x_test), "y": torch.tensor(y_test)},
    }


def get_wedge_data(num_points: int, noise: float = 0.0, test_data_fraction: float = 0.2):
    # Create base x values
    x = torch.linspace(-1, 1, num_points).unsqueeze(1)
    
    # Create wedge shape (^) with absolute value
    y = (1 - torch.abs(x) * 2) + torch.randn_like(x) * noise
    
    # Normalize to range [-0.5, 0.5]
    x = x - x.min()
    y = y - y.min()
    x = x / x.max() - 0.5
    y = y / y.max() - 0.5
    
    # Split into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(x.numpy(), y.numpy(), test_size=test_data_fraction, random_state=0)
    
    return {
        "train": {"x": torch.tensor(x_train), "y": torch.tensor(y_train)},
        "test": {"x": torch.tensor(x_test), "y": torch.tensor(y_test)},
    }

def get_x_square_data(num_points: int, noise: float = 0.0, test_data_fraction: float = 0.2):
    x = torch.linspace(-1, 1, num_points).unsqueeze(1)  # Add batch dimension
    y = (x ** 2 + torch.randn_like(x) * noise)  # Add batch dimension
    
    x_train, x_test, y_train, y_test = train_test_split(x.numpy(), y.numpy(), test_size=test_data_fraction, random_state=0)
    
    return {
        "train": {"x": torch.tensor(x_train), "y": torch.tensor(y_train)},
        "test": {"x": torch.tensor(x_test), "y": torch.tensor(y_test)},
    }

def get_x_sin_x_data(num_points: int, noise: float = 0.0, test_data_fraction: float = 0.2):
    x = torch.linspace(-1, 1, num_points).unsqueeze(1)  # Add batch dimension
    y = torch.where(
        x != 0, 
        x * torch.sin(1 / x) + torch.randn_like(x) * noise, 
        torch.tensor(0.0)  # Define f(0) = 0 explicitly
    )
    
    x_train, x_test, y_train, y_test = train_test_split(x.numpy(), y.numpy(), test_size=test_data_fraction, random_state=0)
    
    return {
        "train": {"x": torch.tensor(x_train, dtype=torch.float32), "y": torch.tensor(y_train, dtype=torch.float32)},
        "test": {"x": torch.tensor(x_test, dtype=torch.float32), "y": torch.tensor(y_test, dtype=torch.float32)},
    }

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from sklearn.model_selection import train_test_split

def get_drawn_data(test_data_fraction: float = 0.2):
    """Allows the user to draw a dataset by clicking and dragging."""
    drawn_points = []
    drawing = False  # Flag to indicate when the user is drawing

    def onpress(event):
        nonlocal drawing
        if event.inaxes:
            drawing = True

    def onrelease(event):
        nonlocal drawing
        drawing = False

    def onmotion(event):
        if drawing and event.xdata is not None and event.ydata is not None:
            drawn_points.append((event.xdata, event.ydata))
            ax.scatter(event.xdata, event.ydata, c='r', s=5)
            plt.draw()

    fig, ax = plt.subplots()
    ax.set_title("Draw your dataset. Click and drag to draw. Close the window when done.")
    plt.ylim(-1, 1)
    plt.xlim(-1, 1)
    fig.canvas.mpl_connect('button_press_event', onpress)
    fig.canvas.mpl_connect('button_release_event', onrelease)
    fig.canvas.mpl_connect('motion_notify_event', onmotion)
    plt.show()
    
    if len(drawn_points) < 2:
        raise ValueError("Not enough points drawn to create a dataset.")
    
    drawn_points = np.array(drawn_points)
    x, y = drawn_points[:, 0], drawn_points[:, 1]
    
    # Normalize data to be in range [-0.5, 0.5]
    x = x - x.min()
    y = y - y.min()
    x = x / x.max() - 0.5
    y = y / y.max() - 0.5
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_data_fraction, random_state=0)
    
    return {
        "train": {"x": torch.tensor(x_train, dtype=torch.float32).unsqueeze(1), 
                   "y": torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)},
        "test": {"x": torch.tensor(x_test, dtype=torch.float32).unsqueeze(1), 
                  "y": torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)},
    }


dataset_map = {
    "sine_wave": get_sine_wave_data,
    "line": get_line_data,
    "line_through_zero": get_line_through_zero_data,
    "zigzag_line": get_zigzag_line_data,
    "x_square": get_x_square_data,
    "wedge": get_wedge_data,
}

def get_dataset(name: str, num_points: int, noise: float = 0.0, test_data_fraction: float = 0.2):
    assert name in dataset_map, f"Dataset {name} not found! Please choose from {list(dataset_map.keys())}"
    return dataset_map[name](num_points, noise, test_data_fraction)