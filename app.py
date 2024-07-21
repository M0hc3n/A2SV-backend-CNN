from flask import jsonify, Flask, request, make_response

from utils.parse_file import get_path_from_file
from utils.load_model import load_model
from utils.preprocesser import preprocess_image
from utils.prediction import predict
from utils.response_handler import get_response_from_model_output

from utils.llama_handler import ask_llm_handler, get_tip_from_llm

from errors.non_formatted_input import input_non_valid_error
from errors.status import status

import requests

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

    llm_response = ask_llm_handler(False, response=response)

    return make_response(
        jsonify({"data": {**response, "llm_response": llm_response}}), status["success"]
    )


@app.route("/diagnose", methods=["POST"])
def diagnose():
    try:
        data = request.json
        symptoms = data.get("symptoms")

    except KeyError:
        return make_response(jsonify(input_non_valid_error), status["input_not_valid"])

    response = ask_llm_handler(True, symptoms=symptoms)

    return make_response(
        jsonify({"data": {"llm_response": response}}), status["success"]
    )


@app.route("/tip", methods=["GET"])
def get_tip():
    response = get_tip_from_llm()

    return make_response(
        jsonify({"data": {"llm_response": response}}), status["success"]
    )


if __name__ == "__main__":
    app.run(debug=True, port=3000, host="0.0.0.0")
