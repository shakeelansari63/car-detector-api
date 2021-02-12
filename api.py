from flask import Flask, jsonify, make_response, request
from car_detection import predict as predict_cars
import os

app = Flask(__name__)


@app.route('/')
def root():
    response = {
        'message': 'This is Car Detection API'
    }
    return jsonify(response), 200


@app.route('/predict')
def predict():
    img = request.args.get('img')

    if not img:
        return jsonify({"message": "Image missing "}), 400
    else:
        try:
            pred = predict_cars(img)
            return jsonify({"message": "Successful", "prediction": pred}), 200
        except Exception as e:
            print(e)
            return jsonify({"message": "Image not found"}), 404


if __name__ == "__main__":
    app.run(host='0.0.0.0', port='8080', debug=True)
