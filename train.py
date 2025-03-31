import torch.nn as nn
from compneuro.training import train_model
from compneuro.mlp import build_mlp_model

# Create results directory
save_dir = "results/vis/single_neuron"

visualize_every_nth_step = 1
num_epochs = 100
batch_size = 32 

# Initialize model
model = build_mlp_model(size_sequence=[1, 20, 10, 1], activation=nn.LeakyReLU(), bias=True)

test_loss = train_model(
    model=model,
    dataset_name="drawn",
    learning_rate=0.01,
    batch_size=batch_size,
    num_epochs=num_epochs,
    num_data_points=100,
    dataset_noise=0.1,
    test_data_fraction=0.3,
    visualize_every_nth_step=visualize_every_nth_step,
    video_frames_folder=save_dir,
    save_video_as="single_neuron_training.mp4",
    video_fps=30
)