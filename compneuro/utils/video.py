import imageio
from PIL import Image
import numpy as np


def generate_video(
    list_of_pil_images: list, framerate: int, filename: str, size: tuple
):
    # Create a list to store the resized images
    resized_images = []

    # Iterate over each PIL image
    for pil_image in list_of_pil_images:
        # Resize the image
        resized_image = pil_image.resize(size, Image.NEAREST)

        # Convert the image to numpy array
        resized_array = np.array(resized_image)

        # Append the resized image to the list
        resized_images.append(resized_array)

    # Save the resized images as a video using imageio
    imageio.mimwrite(filename, resized_images, fps=framerate)