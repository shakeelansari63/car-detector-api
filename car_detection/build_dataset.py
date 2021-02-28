import os
from cv2 import cv2 as cv
from .image_ops import IMG
from .manage_labels import write_labels


def build_dataset(src_dir):
    """ """
    # Set directories
    tgt_dir = os.path.dirname(os.path.abspath(__file__))
    train_dir = os.path.join(tgt_dir, 'train')
    test_dir = os.path.join(tgt_dir, 'test')

    # Make directroies if not exist
    if not os.path.isdir(train_dir):
        os.mkdir(train_dir)
    if not os.path.isdir(test_dir):
        os.mkdir(test_dir)

    # Read source directory and get all labels
    src_label_dir = os.listdir(src_dir)

    # Lets make list if labels
    labels = []
    for label_dir in src_label_dir:
        if not label_dir.startswith('.'):
            # Add this to labels list
            print("Add Label {} to list".format(label_dir))
            labels.append(label_dir)

            # Setup label variables
            src_label_path = os.path.join(src_dir, label_dir)
            train_label_path = os.path.join(train_dir, label_dir)
            test_label_path = os.path.join(test_dir, label_dir)

            # Crete target label directories if they don't exist
            if not os.path.isdir(train_label_path):
                os.mkdir(train_label_path)
            if not os.path.isdir(test_label_path):
                os.mkdir(test_label_path)

            # Read all images from source dir
            image_list = os.listdir(src_label_path)
            train_images = image_list[50:]
            test_images = image_list[:50]

            # Loop over Train Images and write to train folder
            for seq, data_image in enumerate(train_images):
                if data_image.lower().endswith('.jpg') or data_image.lower().endswith('.jpeg'):
                    # Read image in CV
                    img = os.path.join(src_label_path, data_image)
                    cv_im = IMG(img)

                    # Convert and write image to target
                    tgt = os.path.join(train_label_path, str(seq) + '.jpg')
                    cv_im.convert_for_tf(tgt)

            # Loop over Test umages and write to test folder
            for seq, data_image in enumerate(test_images):
                if data_image.lower().endswith('.jpg') or data_image.lower().endswith('.jpeg'):
                    # Read image in CV
                    cv_im = IMG(os.path.join(src_label_path, data_image))

                    # Convert and write image to target
                    tgt = os.path.join(test_label_path, str(seq) + '.jpg')
                    cv_im.convert_for_tf(tgt)

    # Write the Lables in CSV file for maintaining Sequence for future modules
    write_labels(labels)
