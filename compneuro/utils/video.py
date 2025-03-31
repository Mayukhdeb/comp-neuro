import imageio
from PIL import Image
import numpy as np


def generate_video(
    list_of_pil_images: list, framerate: int, filename: str, size: tuple = None
):
    # Create a list to store the resized images
    resized_images = []

    # Iterate over each PIL image
    for pil_image in list_of_pil_images:
        if size is not None:
            # Resize the image
            resized_image = pil_image.resize(size, Image.NEAREST)
        else:
            resized_image = pil_image

        # Convert the image to numpy array
        resized_array = np.array(resized_image)

        # Append the resized image to the list
        resized_images.append(resized_array)

    # Save the resized images as a video using imageio
    imageio.mimwrite(filename, resized_images, fps=framerate)

from IPython.display import HTML
from base64 import b64encode
def show_video(filename, width=200):
    mp4 = open(filename, 'rb').read()
    data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
    return HTML(f"""
    <video width="{width}" controls>
        <source src="{data_url}" type="video/mp4">
    </video>
    """)