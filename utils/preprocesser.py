import keras.utils as image
import numpy as np

from utils.config.model_config import model_config

def preprocess_image(path):
    img = image.load_img(path, target_size=model_config["target_size"])
    img = np.array(img)
    img = img.reshape(model_config["reshape_size"])
    img = img / 255

    return img