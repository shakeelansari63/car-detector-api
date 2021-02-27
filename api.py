# Cheroot for production ready WSGI server
from cheroot.wsgi import Server as WSGIServer
# Flask app requires
from flask import Flask, jsonify, request
from car_detection import predict as predict_cars, load_classifier
import os
# Datetime module for builing file name
import datetime
import random
# Enable Logging before creating app
import logging


# Logging Config
logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s',
                    level=logging.DEBUG)

# Make Flask App
app = Flask(__name__)
app.config['image_dir'] = 'images'
app.config['allowed_files'] = ['jpg', 'jpeg']
app.config['classifier'] = load_classifier()


#######################################
#          Helper Functions           #
#######################################
def generate_file_name():
    now = datetime.datetime.now()
    formatted_date = now.strftime("%Y%m%d%H%M%S%f")

    random_seed = str(random.randint(1, 99999)).zfill(5)

    tgt_file_id = formatted_date + random_seed
    tgt_file = os.path.join(app.config['image_dir'],
                            tgt_file_id + '.jpg')

    return tgt_file


#######################################
#              Routes                 #
#######################################
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
@app.route('/predict', methods=['POST'])
def predict():
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

    # Genereate new target file name
    img = generate_file_name()

    # Write the file on server
    file.save(img)

    # Check for prediction flag in request
    if os.path.isfile(img):
        # Predict image
        try:
            pred = predict_cars(img, app.config['classifier'])

            # Delete files after prediction
            os.remove(img)

            return jsonify({"message": "Successfully Predicted",
                            "prediction": pred}), 200

        # Return error for files which cannot be read as image array
        except Exception as e:
            logging.error(e)
            return jsonify({"message": "Invalid image file"}), 400

    else:
        return jsonify({"message": "Unable to upload file"}), 500


if __name__ == "__main__":
    #app.run(host='0.0.0.0', port='8080', debug=True)
    server = WSGIServer(('0.0.0.0', 8080), app)
    try:
        logging.info('Server Starting')
        server.start()
    except Exception as e:
        logging.error(e)
        server.stop()
        logging.error('Server Stopped')
