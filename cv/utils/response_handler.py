import numpy as np

from utils.config.model_label import skin_labels
from utils.config.medicines import find_medicine

def get_response_from_model_output(prediction):
    pred = np.argmax(prediction)

    disease = skin_labels[pred]
    accuracy = prediction[0][pred]
    accuracy = round(accuracy * 100, 2)
    medicine = find_medicine(pred)

    return {
        "disease": disease,
        "accuracy": accuracy,
        "medicine": medicine,
    }
