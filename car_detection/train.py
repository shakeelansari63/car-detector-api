# Training The Tensorflow model for car detection
# Import OpenCV for reading images
from cv2 import cv2 as cv
# Keras Image processing libraries to convert image to array
from keras.preprocessing.image import img_to_array
# Import Keras Sequential Model
from keras.models import Sequential
# Import Layeers needed for CNN
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
# Numpy Utils from Keras
from keras.utils import np_utils
# Os library for lsiting files
import os
# Import Numpy for data reshaping
import numpy as np
# Labels
from .manage_labels import read_labels


def train_model():
    ##############################################
    #   Set Variables
    ##############################################
    code_dir = os.path.dirname(os.path.abspath(__file__))
    train_dir = os.path.join(code_dir, 'train')
    test_dir = os.path.join(code_dir, 'test')

    train_features = []
    train_labels = []
    test_features = []
    test_labels = []

    img_rows = 60
    img_cols = 60
    img_clrs = 3

    labels = read_labels()
    print(labels)

    ##############################################
    #   Gererate Sets for Training Models
    ##############################################
    # Training Features and Labels for cars
    for lbl_id, label in enumerate(labels):
        train_feature_dir = os.path.join(train_dir, label)
        test_feature_dir = os.path.join(test_dir, label)

        print("Reading features from {}".format(label))

        # Read all images from Training dir
        for fl in os.listdir(train_feature_dir):
            # Read image file
            img = cv.imread(os.path.join(train_feature_dir, fl))
            # Convert image to array
            feature = img_to_array(img)
            # Add features to training set
            train_features.append(feature)
            # Add Label as car to training set
            train_labels.append(lbl_id)

        # Read all images from Testing dir
        for fl in os.listdir(test_feature_dir):
            # Read image file
            img = cv.imread(os.path.join(test_feature_dir, fl))
            # Convert image to array
            feature = img_to_array(img)
            # Add features to training set
            test_features.append(feature)
            # Add Label as car to training set
            test_labels.append(lbl_id)

    ##############################################
    #   Lets see what we have till now
    ##############################################
    train_data_size = len(train_features)
    test_data_size = len(test_features)
    print('Training Dataset: {}'.format(train_data_size))
    print('Testing Dataset: {}'.format(test_data_size))

    ##############################################
    #   Reshape the data for consistency
    ##############################################
    train_features = np.array(train_features)
    train_features = train_features.reshape((train_data_size,
                                             img_rows,
                                             img_cols,
                                             img_clrs))
    train_features = train_features.astype('float32')
    train_features /= 255

    train_labels = np.array(train_labels)
    train_labels = train_labels.reshape(train_data_size, 1)
    train_labels = np_utils.to_categorical(train_labels, 2)

    test_features = np.array(test_features)
    test_features = test_features.reshape((test_data_size,
                                           img_rows,
                                           img_cols,
                                           img_clrs))
    test_features = test_features.astype('float32')
    test_features /= 255

    test_labels = np.array(test_labels)
    test_labels = test_labels.reshape(test_data_size, 1)
    test_labels = np_utils.to_categorical(test_labels, 2)

    ##############################################
    #   Building the classifier
    ##############################################
    classifier = Sequential()
    # Step 1: Add Convolution Layer
    classifier.add(Conv2D(
        filters=32,
        kernel_size=(3, 3),
        input_shape=(img_rows, img_cols, img_clrs),
        activation='relu'
    ))

    # Step 2: Add another Convolution Layer
    classifier.add(Conv2D(
        filters=32,
        kernel_size=(3, 3),
        activation='relu'
    ))

    # Step 3: Pooling filter
    classifier.add(MaxPooling2D(
        pool_size=(2, 2)
    ))

    # Step 4: Add Dropout to avoid Overfitting
    classifier.add(Dropout(0.5))

    # Step 5: Flatten the Image to feed to Neural Network
    classifier.add(Flatten())

    # Step 6: Fully Connected Neural Network
    classifier.add(Dense(
        units=128,
        activation='relu'
    ))

    # Step 7: Fully Connected Neural Network for output
    classifier.add(Dense(
        units=21,
        activation='softmax'
    ))

    ##############################################
    #   Compile the classifier
    ##############################################
    classifier.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    ##############################################
    #   Fit Model to Training set
    ##############################################
    classifier.fit(
        train_features,
        train_labels,
        batch_size=10,
        epochs=25,
        validation_split=0.1,
        verbose=1
    )

    ##############################################
    #   Evaluate Model on Testing set
    ##############################################
    _, score = classifier.evaluate(
        test_features,
        test_labels,
        verbose=1
    )

    print('Model accuracy: {}'.format(score))

    ##############################################
    #   Save the h5 Model
    ##############################################
    classifier.save(os.path.join(code_dir, 'car_detection_using_cnn.h5'))


if __name__ == '__main__':
    train_model()
