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

    path = get_path_from_file(file)

    model = load_model()

    img = preprocess_image(path)

    prediction = predict(model, img)

    response = get_response_from_model_output(prediction)

    return make_response(jsonify(response), status["success"])


if __name__ == "__main__":
    app.run(debug=True, port=3000)
