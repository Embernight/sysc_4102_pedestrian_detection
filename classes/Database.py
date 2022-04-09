import csv
import os

import numpy as np


class Database:
    """ A class to manage a database for use with TensorFlow.

    Attributes
    ----------
    annotation_dir: str
        Annotation file directory
    video_image_dir: str
        Video image file directory
    test: list
        Files to use for testing.
    train: list
        Files to use for training.
    validate: list
        Files to use for validation.
    """

    def __init__(self, annotation_dir, video_image_dir):
        """Constructs the necessary attributes for the database object.

        :param annotation_dir: Annotation file directory.
        :param video_image_dir: Video image file directory.
        """
        if not os.path.isdir(video_image_dir):
            print("Video image directory does not exist")
            return
        if not os.path.isdir(annotation_dir):
            print("Annotation directory does not exist")
            return

        self.annotation_dir = annotation_dir
        self.video_image_dir = video_image_dir
        self.annotations = {}
        self.test = []
        self.train = []
        self.validate = []

    def generate_data_split(self,
                            percent_test=10,
                            percent_train=80,
                            percent_validate=10,
                            load_from_files=False,
                            test_file="./datasets/test.csv",
                            train_file="./datasets/train.csv",
                            validate_file="./datasets/validate.csv",
                            output_to_files=False):
        """Splits your data set into separate data set for testing, training, and validating.

        :param percent_test: Percentage of files for testing.
        :param percent_train: Percentage of files for training.
        :param percent_validate: Percentage of files for validation.
        :param load_from_files: If you already have a split data set load them.
        :param test_file: Location of the test data set.
        :param train_file: Location of the train data set.
        :param validate_file: Location of the validation data set.
        :param output_to_files: True if you want to save the output, False otherwise
        :return:
        """
        # Loads the pre-split data sets.
        if load_from_files:
            if not os.path.isfile(test_file) and not os.path.isfile(train_file) and not os.path.isfile(validate_file):
                print(f"Check dataset file paths:\n{test_file}\n{train_file}\n{validate_file}")
            with open(test_file, "r") as test:
                r = csv.reader(test)
                for line in r:
                    if len(line) != 0:
                        self.test.append(line[0])
            with open(train_file, "r") as train:
                r = csv.reader(train)
                for line in r:
                    if len(line) != 0:
                        self.train.append(line[0])
            with open(validate_file, "r") as validate:
                r = csv.reader(validate)
                for line in r:
                    if len(line) != 0:
                        self.validate.append(line[0])
            return

        if (percent_test + percent_train + percent_validate) != 100:
            print("Test, Train, and Validate Percentage must add up to 100")
            return

        # The follow lines read the image directory and splits the data
        # into test, train, and validate data sets.
        images = os.listdir(self.video_image_dir)
        images_len_test = int(len(images) * (percent_test / 100))
        images_len_train = int(len(images) * (percent_train / 100))

        increment = len(images) / images_len_test
        for i in np.arange(0, len(images), increment):
            self.test.append(images[int(i)])

        image_diff_test = list(set(images) - set(self.test))
        increment = len(image_diff_test) / images_len_train
        for i in np.arange(0, len(image_diff_test), increment):
            self.train.append(image_diff_test[int(i)])

        self.validate = list(set(image_diff_test) - set(self.train))

        # Sort the datasets
        self.test.sort()
        self.train.sort()
        self.validate.sort()

        if output_to_files:
            with open(test_file, "w") as test:
                csv.writer(test, delimiter='\n').writerow(self.test)
            with open(train_file, "w") as train:
                csv.writer(train, delimiter='\n').writerow(self.train)
            with open(validate_file, "w") as validate:
                csv.writer(validate, delimiter='\n').writerow(self.validate)

    def pair_annotations_to_images(self):
        """Creates a dictionary of lists containing annotations

        The dictionary can be quarried by the image name to get that images
        respective annotations.
        :return:
        """
        images = os.listdir(self.video_image_dir)
        image_numbers = {}
        annotations = []

        for file in images:
            self.annotations[file] = []
            image_numbers[int(file.replace(".jpg", "")) - 1] = file

        for file in os.listdir(self.annotation_dir):
            if file.endswith(".csv"):
                with open(self.annotation_dir + "/" + file, "r") as f:
                    annotations = list(csv.reader(f))

        for anno in annotations:
            image_filename = image_numbers[int(anno[0]) - 1]
            self.annotations[image_filename].append(anno[1:])
