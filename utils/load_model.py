from keras.models import model_from_json
from utils.config.model_config import model_config


def load_model():
    j_file = open(model_config["model_json_name"], "r")
    loaded_json_model = j_file.read()
    j_file.close()
    model = model_from_json(loaded_json_model)
    model.load_weights(model_config["model_dump"])

    return model
