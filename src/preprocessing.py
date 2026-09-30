"""
This file converts grayscale OT image to RGB image
This file is called when the api endpoint is requested with an OT image
This file prepares the input image suitable to be processed by the trained CNN encoder
"""

import numpy as np
import torch
from PIL import Image

image_size = 50

def preprocess_image(image: Image.Image) -> torch.Tensor:
    image = image.convert("RGB")
    image = image.resize((image_size, image_size))
    image_array = np.asarray(image)
    image_array = image_array.astype(np.float32)
    image_array = image_array / 255.0
    image_tensor = torch.tensor(image_array, dtype = torch.float32)
    # convert (height, width, channels) --> (channels, height, width)
    image_tensor = image_tensor.permute(2, 0, 1)
    # add batch dimension to change  [3, 50, 50] to  [1, 3, 50, 50]
    image_tensor = image_tensor.unsqueeze(0)
    return image_tensor


