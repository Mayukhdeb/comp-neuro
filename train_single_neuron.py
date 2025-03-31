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


def train_model(
    model,
    dataset_name: str,
    learning_rate=0.01,
    batch_size=32,
    num_epochs=100,
    num_data_points=200,
    dataset_noise=0.0,
    test_data_fraction=0.3,
    visualize_every_nth_step=1,
    video_frames_folder=None,
    save_video_as = None,
    video_fps=32
) -> float:
    """
    Trains a model on the given dataset.

    Parameters:
    - model: PyTorch model to be trained
    - dataset: Dictionary containing training and test data (keys: 'train' and 'test')
    - learning_rate: Learning rate for the optimizer
    - batch_size: Batch size for training
    - num_epochs: Number of epochs to train
    - visualize_every_nth_step: Frequency of visualization
    - video_frames_folder: Directory to save visualizations (if None, visualizations are not saved)
    - save_video_as: Path to save the video (if None, video is not saved)
    - video_fps: Frames per second for the video

    Returns:
    - Final test loss (float)
    """
    if video_frames_folder is not None:
        os.system(f"rm -rf {video_frames_folder} && mkdir -p {video_frames_folder}")

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    dataset = get_dataset(
        name=dataset_name,
        num_points=num_data_points,
        test_data_fraction=test_data_fraction,
        noise=dataset_noise
    )
    
    train_x, train_y = dataset["train"]["x"], dataset["train"]["y"]
    test_x, test_y = dataset["test"]["x"], dataset["test"]["y"]

    train_dataset = TensorDataset(train_x, train_y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    step = 0
    video_frame_filenames = []
    
    for epoch in tqdm(range(num_epochs), desc="Training"):
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            
            if video_frames_folder is not None and step % visualize_every_nth_step == 0:
                with torch.no_grad():
                    test_preds = model(test_x)
                    fig = plt.figure(figsize=(12, 8))
                    plt.scatter(train_x.numpy(), train_y.numpy(), label="Train data", color="gray", alpha=0.3)
                    plt.scatter(test_x.numpy(), test_y.numpy(), label="Test data", color="red", linewidth=2)
                    plt.scatter(test_x.numpy(), test_preds.numpy(), label="Predicted", color="blue", linewidth=2)
                    plt.xlabel("Input")
                    plt.ylabel("Output")
                    plt.legend()
                    plt.tight_layout()
                    ## remove upper and right spine
                    plt.gca().spines['top'].set_visible(False)
                    plt.gca().spines['right'].set_visible(False)
                    plt.title(f"Epoch {step}: Loss = {loss.item():.4f}")
                    filename = os.path.join(save_dir, f"epoch_{step:04d}.png")
                    fig.savefig(filename)
                    plt.close()
                    video_frame_filenames.append(filename)
            
            step += 1
    
    # Compute final test loss
    with torch.no_grad():
        final_test_preds = model(test_x)
        final_test_loss = criterion(final_test_preds, test_y).item()

    if video_frames_folder is not None and save_video_as is not None:
        generate_video(
            list_of_pil_images=[Image.open(filename) for filename in video_frame_filenames],
            framerate=video_fps,
            size=(600, 400),
            filename=save_video_as
        )
    
    return final_test_loss


# Create results directory
save_dir = "results/vis/single_neuron"


visualize_every_nth_step = 1
num_epochs = 100
batch_size = 32 

# Initialize model
model = build_mlp_model(size_sequence=[1, 50, 8, 1], activation=nn.LeakyReLU())  # Using Tanh for sine fitting

test_loss = train_model(
    model=model,
    dataset_name="sine_wave", ## options: "sine_wave", "line", "zigzag_line"
    learning_rate=0.01,
    batch_size=batch_size,
    num_epochs=num_epochs,
    num_data_points=200,
    dataset_noise=0.1,
    test_data_fraction=0.3,
    visualize_every_nth_step=visualize_every_nth_step,
    video_frames_folder=save_dir,
    save_video_as="single_neuron_training.mp4",
    video_fps=30
)