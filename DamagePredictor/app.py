from flask import Flask, request, jsonify
from machine_learning_service import predict_classifications, train_model, delete_model

app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict():
    """
    The predict endpoint to predict classifications.

    :return: The predictions.
    """

    if "test_values" in request.files:
        try:
            test_values = request.files["test_values"]
            predictions = predict_classifications(test_values)
            return jsonify({"predictions": predictions.tolist()})
        except Exception as e:
            print(str(e))
            return print(str(e))
    else:
        return "Error: No file attached"


@app.route("/train", methods=["POST"])
def train():
    """
    The train endpoint to train/retrain a model.

    :return: The string status indicating success or failure.
    """

    if request.files:
        try:
            train_values = request.files["train_values"]
            train_labels = request.files["train_labels"]

            train_model(train_values, train_labels)
            return "Success"
        except Exception as e:
            print(str(e))
            return "Error: Failed to train model"


@app.route("/delete", methods=["DELETE"])
def delete():
    """
    The delete endpoint to delete an existing model.

    :return: The string status indicating success or failure.
    """
    try:
        delete_model()
        return "Model deleted"
    except Exception as e:
        print(str(e))
        return "Error: Failed to delete model"


if __name__ == "__main__":
    clf = None
    app.run(host="0.0.0.0", port=5000, debug=True)
