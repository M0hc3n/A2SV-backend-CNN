# from utils.config.model_label import skin_labels
# from utils.config.medicines import find_medicine

def get_response_from_model_output(pred_class, pred_idx, outputs):
    # medicine = find_medicine(pred)

    return {
        "disease": pred_class,
        "accuracy": outputs[pred_idx].item(),
        # "medicine": medicine,
    }
