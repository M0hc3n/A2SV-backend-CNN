from flask import jsonify, Flask, request, make_response

from utils.parse_file import get_path_from_file
from utils.load_model import load_model
from utils.preprocesser import preprocess_image
from utils.prediction import predict
from utils.response_handler import get_response_from_model_output

from errors.non_formatted_input import input_non_valid_error
from errors.status import status

app = Flask(__name__)

@app.route("/detect", methods=["POST"])
def detect():
    try:
        file = request.files["file"]
    except KeyError:
        return make_response(jsonify(input_non_valid_error), status["input_not_valid"])

    imageBytesIO = get_path_from_file(file)

    img = preprocess_image(imageBytesIO)

    learn = load_model()

    pred_class, pred_idx, outputs = predict(learn, img)

    response = get_response_from_model_output(pred_class, pred_idx, outputs)
    return response


if __name__ == "__main__":
    app.run(debug=True, port=3000, host="0.0.0.0")
