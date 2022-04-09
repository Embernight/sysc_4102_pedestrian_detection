import os
import sys

import cv2 as cv
import numpy as np

from classes.Database import Database
from classes.Rect import Rect
from classes.edge_detection_model import edge_detection


def create_database() -> Database:
    """Creates a database for use in the edge detection class.

    :return: Database
    """
    db = Database("./annotations", "./images")
    db.generate_data_split(load_from_files=True)
    db.pair_annotations_to_images()
    return db


def train(database, height, width):
    """Used to initialize the training function of the edge detection class."""
    ed = edge_detection(database)
    ed.generate_xy_data([height, width], set_type="train")
    ed.train_model(epochs=20)


def validate(database, height, width):
    """Used to initialize the validation function of the edge detection class."""
    ed = edge_detection(database)
    ed.train_model(load_model=True)
    ed.generate_xy_data([height, width], set_type="validate")
    ed.model.evaluate(ed.validate_x, ed.validate_y, verbose=2)


def test(height, width):
    """Used to initialize the highlight people function of the edge detection class."""
    ed = edge_detection(Database("./annotations", "./images"))
    ed.train_model(load_model=True)
    ed.highlight_people("./test_images", [height, width])


def show_steps(save=False):
    """Generates images for each of the steps required to extract features from the image.

    :param save: Determines if the output should be saved or shown.
    """
    images = os.listdir("./test_images")
    for im in images:
        # Used to generate the feature extraction process.
        image = cv.imread("./test_images" + "/" + im)
        grey = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
        blur = cv.GaussianBlur(grey, (3, 3), 0)
        hist = cv.equalizeHist(blur)
        edges = cv.Canny(hist, 100, 200)
        contours, _ = cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        # Draws the contours on an image
        image_contours = image.copy()
        cv.drawContours(image_contours, contours, -1, (255, 0, 0), 3)

        # Convert contours to bounding rectangles and draws them on an image.
        image_contour_bounds = image.copy()
        rectangles = []
        for cnt in contours:
            x, y, w, h = cv.boundingRect(cnt)
            rectangles.append(Rect(x, y, w, h))
            cv.rectangle(image_contour_bounds, (x, y), (x + w, y + h), (255, 0, 0), 2)

        bounds = Rect.group_rectangles(rectangles)

        # Draws the merged bounding rectangles on an image.
        image_merge = image.copy()
        b: Rect
        for b in bounds:
            cv.rectangle(image_merge, (b.x, b.y), (b.x + b.w, b.y + b.h), (255, 0, 0), 2)

        # Displays or saves the images.
        if save:
            cv.imwrite("./steps/original.jpg", image)
            cv.imwrite("./steps/grey.jpg", grey)
            cv.imwrite("./steps/blur.jpg", blur)
            cv.imwrite("./steps/equalize.jpg", hist)
            cv.imwrite("./steps/edges.jpg", edges)
            cv.imwrite("./steps/contours.jpg", image_contours)
            cv.imwrite("./steps/contours_bounds.jpg", image_contour_bounds)
            cv.imwrite("./steps/merger.jpg", image_merge)
        else:
            cv.imshow("original", image)
            cv.imshow("grey", grey)
            cv.imshow("blur", blur)
            cv.imshow("equalize", hist)
            cv.imshow("edges", edges)
            cv.imshow("contours", image_contours)
            cv.imshow("contours_bounds", image_contour_bounds)
            cv.imshow("merger", image_merge)
            cv.waitKey()


def create_collage():
    """Creates a collage from the generated images."""
    image = cv.imread("./steps/original.jpg")
    grey = cv.imread("./steps/grey.jpg")
    blur = cv.imread("./steps/blur.jpg")
    hist = cv.imread("./steps/equalize.jpg")
    edges = cv.imread("./steps/edges.jpg")
    image_contours = cv.imread("./steps/contours.jpg")
    image_contour_bounds = cv.imread("./steps/contours_bounds.jpg")
    image_merge = cv.imread("./steps/merger.jpg")
    detect_image = cv.imread("./steps/detect.jpg")

    cv.putText(image, "Original", (25, 50), cv.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2, cv.LINE_AA)
    cv.putText(grey, "Grey", (25, 50), cv.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2, cv.LINE_AA)
    cv.putText(blur, "Blur", (25, 50), cv.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2, cv.LINE_AA)
    cv.putText(hist, "Equalize", (25, 50), cv.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2, cv.LINE_AA)
    cv.putText(edges, "Edges", (25, 50), cv.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2, cv.LINE_AA)
    cv.putText(image_contours, "Contours", (25, 50), cv.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2, cv.LINE_AA)
    cv.putText(image_contour_bounds, "Contours Bounds", (25, 50), cv.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2, cv.LINE_AA)
    cv.putText(image_merge, "merge Bounds", (25, 50), cv.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2, cv.LINE_AA)
    cv.putText(detect_image, "Detect", (25, 50), cv.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2, cv.LINE_AA)

    row1 = np.hstack([image, grey, blur])
    row2 = np.hstack([hist, edges, image_contours])
    row3 = np.hstack([image_contour_bounds, image_merge, detect_image])

    collage = np.vstack([row1, row2, row3])
    cv.imwrite("process_collage.png", collage)


def handle_args():
    """Used to handle call arguments"""
    if len(sys.argv) != 2:
        print("You must only provide one argument.\n"
              "Supported Calls:\n"
              "python main.py test\n"
              "python main.py train\n"
              "python main.py show_steps\n"
              "python main.py save_steps\n"
              "python main.py validate\n")
        return 1

    height, width = 200, 100
    if sys.argv[1] == "collage":
        create_collage()
    elif sys.argv[1] == "test":
        test(height, width)
    elif sys.argv[1] == "train":
        train(create_database(), height, width)
    elif sys.argv[1] == "show_steps":
        show_steps()
    elif sys.argv[1] == "save_steps":
        show_steps(save=True)
    elif sys.argv[1] == "validate":
        validate(create_database(), height, width)


if __name__ == '__main__':
    handle_args()
