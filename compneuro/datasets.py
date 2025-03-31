import torch
from sklearn.model_selection import train_test_split

def get_sine_wave_data(num_points: int, noise: float = 0.0, test_size: float = 0.2):
    x = torch.linspace(0, 2 * 3.14159, num_points).unsqueeze(1)  # Add batch dimension
    y = (torch.sin(x) + torch.randn_like(x) * noise)  # Add batch dimension
    
    x_train, x_test, y_train, y_test = train_test_split(x.numpy(), y.numpy(), test_size=test_size, random_state=0)
    
    return {
        "train": {"x": torch.tensor(x_train), "y": torch.tensor(y_train)},
        "test": {"x": torch.tensor(x_test), "y": torch.tensor(y_test)},
    }

def get_line_data(num_points: int, noise: float = 0.0, test_size: float = 0.2):
    x = torch.linspace(0, 10, num_points).unsqueeze(1)  # Add batch dimension
    y = (2 * x + 1 + torch.randn_like(x) * noise)  # Add batch dimension
    
    x_train, x_test, y_train, y_test = train_test_split(x.numpy(), y.numpy(), test_size=test_size, random_state=0)
    
    return {
        "train": {"x": torch.tensor(x_train), "y": torch.tensor(y_train)},
        "test": {"x": torch.tensor(x_test), "y": torch.tensor(y_test)},
    }

def get_zigzag_line_data(num_points: int, noise: float = 0.0, test_size: float = 0.2):
    x = torch.linspace(-1, 1, num_points).unsqueeze(1)  # Add batch dimension
    y = (torch.abs((x * 4 % 4) - 2) * 5 + torch.randn_like(x) * noise) # Add batch dimension
    
    x = x - x.min()
    y = y - y.min()

    x = x/x.max()
    y = y/y.max()
    x = x - 0.5
    y = y - 0.5

    x_train, x_test, y_train, y_test = train_test_split(x.numpy(), y.numpy(), test_size=test_size, random_state=0)
    
    return {
        "train": {"x": torch.tensor(x_train), "y": torch.tensor(y_train)},
        "test": {"x": torch.tensor(x_test), "y": torch.tensor(y_test)},
    }


dataset_map = {
    "sine_wave": get_sine_wave_data,
    "line": get_line_data,
    "zigzag_line": get_zigzag_line_data,
}

def get_dataset(name: str, num_points: int, noise: float = 0.0, test_size: float = 0.2):
    assert name in dataset_map, f"Dataset {name} not found! Please choose from {list(dataset_map.keys())}"
    return dataset_map[name](num_points, noise, test_size)