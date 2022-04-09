import os

import cv2 as cv
import numpy as np
import tensorflow as tf

from sklearn.utils import shuffle
from classes.Rect import Rect


class edge_detection:
    """ A class that contains the information required to train, validate, and test a model.

    Attributes
    ----------
    database: Database
        Database containing the annotations, images, and data set splits.
    test_x: float[]
        Image splice to be used in model testing.
    test_y: unint8[]
        Truth values for the test set.
    train_x: float[]
        Image splice to be used in model training.
    train_y: unint8[]
        Truth values for the train set.
    validate_x: float[]
        Image splice to be used in model validation.
    validate_y: unint8[]
        Truth values for the validation set.
    model: tf.keras.models.Sequential
        A TensorFlow Keras sequential model.
    """
    def __init__(self, database):
        """Constructs the necessary attributes for the edge_detection object.

        :param database: Database containing the annotations, images, and data set splits.
        """
        self.database = database
        self.test_x = np.zeros(2, float)
        self.test_y = np.zeros(2, np.uint8)
        self.train_x = np.zeros(2, float)
        self.train_y = np.zeros(2, np.uint8)
        self.validate_x = np.zeros(2, float)
        self.validate_y = np.zeros(2, np.uint8)
        self.model = tf.keras.models.Sequential()

    def train_model(self, load_model=False, epochs=5):
        """Used to train a TensorFlow Keras model.

        :param load_model: True if you want to load a pretrained model.
        :param epochs: Number of times to iterate through the training data set.
        :return:
        """
        if load_model:
            self.model = tf.keras.models.load_model('./models/edge_detection')
            return

        # Create the model shape.
        [h, w] = self.train_x.shape[1], self.train_x.shape[2]
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(h, w)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(2)
        ])

        # Set the loss function.
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        # Recompile the trainer with new attributes.
        self.model.compile(optimizer='adam',
                           loss=loss_fn,
                           metrics=['accuracy'])

        # Train and save the model.
        self.model.fit(self.train_x, self.train_y, epochs=epochs)
        self.model.save('./models/edge_detection')

    def highlight_people(self, image_dir: str, shape: list):
        """Used to highlight and count the number of people in an image then display the results.

        :param image_dir: Images to highlight and count people in.
        :param shape: Shape of the image [Height, Width].
        :return:
        """
        images = os.listdir(image_dir)
        temp = cv.imread(image_dir + "/" + images[0])
        img_h, img_w = temp.shape[0], temp.shape[1]
        for im in images:
            # Process to find the features of an image.
            image = cv.imread(image_dir + "/" + im)
            gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
            blur = cv.GaussianBlur(gray, (3, 3), 0)
            hist = cv.equalizeHist(blur)
            edges = cv.Canny(hist, 100, 200)
            contours, _ = cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

            # Convert contours to bounding rectangles.
            rectangles = []
            for cnt in contours:
                x, y, w, h = cv.boundingRect(cnt)
                rectangles.append(Rect(x, y, w, h))

            bounds = Rect.group_rectangles(rectangles)

            # Goes through the list of bounding rectangles.
            # If the bound is too small to contain a human it is ignored.
            # If the bound is large enough it checks for a human.
            # If there is a person it highlights them and increments the human counter.
            h, w = shape[0], shape[1]
            b: Rect
            num_people = 0
            for b in bounds:
                if b.area() < 0.3 * h * w:
                    continue
                col, row = b.x, b.y
                if row + h > img_h:
                    row = row - ((row + h) - img_h)
                if col + w > img_w:
                    col = col - ((col + w) - img_h)
                section = [gray[row:row + h, col:col + w]]
                section = np.array(section, float) / 255
                prediction = self.model.predict(section)
                prediction_percent = tf.nn.softmax(prediction).numpy()
                if prediction_percent[0][1] > 0.4:
                    cv.rectangle(image, (col, row), (col + w, row + h), (255, 0, 0), 2)
                    num_people = num_people + 1

            # Add the number of humans found to the image and display it.
            cv.putText(image, str(num_people), (700, 550), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3, cv.LINE_AA)
            cv.imshow("Test Image", image)
            cv.waitKey()

    def generate_xy_data(self, shape, set_type="train"):
        """Generates a par of image splices with their true values.

        :param shape: Shape of the image [Height, Width].
        :param set_type: The type of data set to generate and store.
        :return:
        """
        if set_type == "train":
            image_names = self.database.train
        elif set_type == "test":
            image_names = self.database.test
        elif set_type == "validate":
            image_names = self.database.validate
        else:
            return

        data = self.database.annotations
        x_data = []
        y_data = []
        for im in image_names:
            # Process to find the features of an image.
            image = cv.cvtColor(cv.imread(self.database.video_image_dir + "/" + im), cv.COLOR_RGB2GRAY)
            blur = cv.GaussianBlur(image, (3, 3), 0)
            hist = cv.equalizeHist(blur)
            edges = cv.Canny(hist, 100, 200)
            contours, _ = cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

            # Convert contours to bounding rectangles.
            rectangles = []
            for cnt in contours:
                x, y, w, h = cv.boundingRect(cnt)
                rectangles.append(Rect(x, y, w, h))

            bounds = Rect.group_rectangles(rectangles)
            annotations = data[im]

            # Goes through the list of bounding rectangles.
            # Adds an image splice to the x data and a 0 to the y value
            # If the bounding rectangle overlaps with annotation
            # change the y value to 1.
            # 0 represent no person.
            # 1 represents a person.
            h, w = shape[0], shape[1]
            img_h, img_w = image.shape[0] - 1, image.shape[1] - 1
            b: Rect
            for b in bounds:
                col, row = b.x, b.y
                if row + h > img_h:
                    row = row - ((row + h) - img_h)
                if col + w > img_w:
                    col = col - ((col + w) - img_h)
                temp = image[row:row + h, col:col + w]
                x_data.append(temp)
                y_data.append(0)
                for anno in annotations:
                    ax, ay, aw, ah = int(anno[0]), int(anno[1]), int(anno[2]) - int(anno[0]), int(anno[3]) - int(
                        anno[1])
                    rect = Rect(col, row, w, h)
                    rect_anno = Rect(ax, ay, aw, ah)
                    overlap = Rect.overlap(rect, rect_anno)
                    if overlap > 0.15:
                        y_data[len(y_data) - 1] = 1

        # Shuffles the data set to improve training later on.
        x_data, y_data = shuffle(x_data, y_data, random_state=0)

        if set_type == "train":
            self.train_x = np.array(x_data, float) / 255
            self.train_y = np.array(y_data, np.uint8)
        elif set_type == "test":
            self.test_x = np.array(x_data, float) / 255
            self.test_y = np.array(y_data, np.uint8)
        elif set_type == "validate":
            self.validate_x = np.array(x_data, float) / 255
            self.validate_y = np.array(y_data, np.uint8)
