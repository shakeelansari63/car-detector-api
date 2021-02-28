import os
import csv


def read_labels():
    with open(get_label_file(), 'r') as fp:
        csv_reader = csv.reader(fp)

        for labels in csv_reader:
            return labels


def write_labels(labels):
    with open(get_label_file(), 'w') as fp:
        csv_writer = csv.writer(fp)

        csv_writer.writerow(labels)


def get_label_file():
    my_dir = os.path.dirname(os.path.abspath(__file__))
    file = 'labels.csv'

    return os.path.join(my_dir, file)
