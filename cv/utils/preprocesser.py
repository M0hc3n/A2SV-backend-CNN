# import keras.utils as image
# import numpy as np

# from utils.config.model_config import model_config

from fastai.vision.all import PILImage


def preprocess_image(imageBytesIO):
    img = PILImage.create(imageBytesIO)


    return img