import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import os
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from compneuro.datasets import get_dataset
from compneuro.utils.video import generate_video
from compneuro.mlp import build_mlp_model




# Create results directory
save_dir = "results/vis/single_neuron"
os.system(f"rm -rf {save_dir} && mkdir -p {save_dir}")

visualize_every_nth_step = 1
num_epochs = 100
batch_size = 32 

# Initialize model
model = build_mlp_model(size_sequence=[1, 50,2, 1], activation=nn.LeakyReLU())  # Using Tanh for sine fitting

# Generate dataset
dataset = get_dataset(name = "zigzag_line", num_points=200, test_size=0.3, noise = 0.0)  # More points for better training

# Optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=0.01)  # Adam optimizer for faster convergence
criterion = nn.MSELoss()  # Standard MSE loss for regression tasks

# Use tensors directly
train_x = dataset["train"]["x"]
train_y = dataset["train"]["y"]
test_x = dataset["test"]["x"]
test_y = dataset["test"]["y"]

# Prepare DataLoader
train_dataset = TensorDataset(train_x, train_y)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

video_frame_filenames = []

# Training loop
step = 0
for epoch in tqdm(range(num_epochs), desc="Training"):
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        predictions = model(batch_x)
        loss = criterion(predictions, batch_y)
        loss.backward()
        optimizer.step()

        if step % visualize_every_nth_step == 0:
            with torch.no_grad():
                test_preds = model(test_x)
                fig = plt.figure(figsize=(6, 4))
                plt.scatter(train_x.numpy(), train_y.numpy(), label="Train data", color="gray", alpha=0.3)
                plt.scatter(test_x.numpy(), test_y.numpy(), label="Test data", color="red", linewidth=2)
                plt.scatter(test_x.numpy(), test_preds.numpy(), label="Predicted", color="blue", linewidth=2)
                plt.xlabel("Input")
                plt.ylabel("Output")
                plt.legend()
                # plt.ylim(1.1 * test_y.min(), 1.1*test_y.max())
                plt.title(f"Epoch {step}: Loss = {loss.item():.4f}")
                filename = os.path.join(save_dir, f"epoch_{step:04d}.png")
                fig.savefig(filename)
                plt.close()
                # print(f"Saved {filename}")
                video_frame_filenames.append(filename)

        step += 1

# Generate video
generate_video(
    list_of_pil_images=[Image.open(filename) for filename in video_frame_filenames],
    framerate=30,
    filename="single_neuron_training.mp4",
    size=(600, 400),
)
