# Cheroot for production ready WSGI server
from cheroot.wsgi import Server as WSGIServer
# Flask app requires
from flask import Flask, jsonify, request
from car_detection import predict as predict_cars, load_classifier
import os
# Werkzeug Utils for Securing file name
from werkzeug.utils import secure_filename
# Enable Logging before creating app
import logging
from logging.config import dictConfig

# Logging Config
dictConfig({
    'version': 1,
    'formatters': {'default': {
        'format': '[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
    }},
    'handlers': {'wsgi': {
        'class': 'logging.StreamHandler',
        'stream': 'ext://flask.logging.wsgi_errors_stream',
        'formatter': 'default'
    }},
    'root': {
        'level': 'DEBUG',
        'handlers': ['wsgi']
    }
})

# Make Flask App
app = Flask(__name__)
app.config['image_dir'] = 'images'
app.config['allowed_files'] = ['jpg', 'jpeg']
app.config['classifier'] = load_classifier()


# Error Handler
@app.errorhandler(404)
def route_not_found(e):
    return jsonify({"message": "Invalid route"}), 404


# Root page
@app.route('/')
def root():
    response = {'message': 'This is Car Detection API'}
    return jsonify(response), 200


# Uplaod File
@app.route('/upload', methods=['POST'])
def upload():
    # Check if file is sent with request
    if 'file' not in request.files:
        return jsonify({'message': 'File not posted'}), 400

    file = request.files['file']

    # Check for empty file anme
    if file.filename == '':
        return jsonify({'message': 'File not posted'}), 400

    # Check for file extension
    if file.filename.rsplit('.', 1)[1].lower() not in app.config['allowed_files']:
        return jsonify({'message': 'Invalid file extension'}), 400

    # Secure the target file name
    tgt_file = os.path.join(app.config['image_dir'],
                            secure_filename(file.filename)
                            )

    # Write the file on server
    file.save(tgt_file)

    # Check for prediction flag in request
    if os.path.isfile(tgt_file):
        return jsonify({'message': 'File uploaded successfully',
                        'filename': tgt_file}), 200
    else:
        return jsonify({"message": "Unable to upload file"}), 500


# Prediction route
@app.route('/predict', methods=['GET'])
def predict():
    img = request.args.get('img')

    if not img:
        return jsonify({"message": "Image missing "}), 400
    else:
        try:
            pred = predict_cars(img, app.config['classifier'])
            return jsonify({"message": "Successfully Predicted",
                            "prediction": pred}), 200

        except Exception as e:
            print(e)
            return jsonify({"message": "Image not found"}), 404


if __name__ == "__main__":
    #app.run(host='0.0.0.0', port='8080', debug=True)
    server = WSGIServer(('0.0.0.0', 8080), app)
    try:
        logging.info('Starting the Server')
        server.start()
    except Exception as e:
        logging.error('Server Stopped')
        logging.error(e)
        server.stop()
