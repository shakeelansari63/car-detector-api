# Import Keras model loading method to load h5
from keras.models import load_model
# Keras Image processing libraries to convert image to array
from keras.preprocessing.image import img_to_array
# Image Ops library to compress image
from .image_ops import IMG
# Import sys for commandline args
import sys
import os
# Numpy for Array
import numpy as np


# Function to predict
def predict(img):
    # Set variables
    imrow, imcol, imclrs = 100, 100, 3

    # Check if file exist
    if os.path.isfile(os.path.abspath(img)):

        # Compress image
        img = IMG(os.path.abspath(img))
        img.resize(imrow, imcol)

        # Get array of image
        img_array = img_to_array(img.get_img())

        # Convert it to NP Array
        img_array = np.array(img_array)

        # Reshape to correct Size
        img_array = img_array.reshape((1, imrow, imcol, imclrs))
        img_array = img_array.astype('float32')
        img_array /= 255

        # Load model
        model_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'car_detection_using_cnn.h5'
        )
        classifier = load_model(model_file)

        # Predict using the classifier
        prediction = classifier.predict(img_array)

        max_val = prediction[0].max()
        idx_max_val = np.where(prediction[0] == max_val)

        return 'No Car' if idx_max_val[0] == 0 else 'Car'

    else:
        raise Exception('Image file does not exist {}'.format(img))


if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise Exception('Image File name missing for prediction')
    else:
        img = sys.argv[1]

        # Predict if file exist
        print(predict(img))
