from __future__ import annotations


class Rect:
    """ A class that represents a rectangle.

    Attributes
    ----------
    x: int
        Top left corner's x coordinate.
    y: int
        Top left corner's y coordinate.
    w: int
        Width of the rectangle.
    h: int
        Height of the rectangle.
    """

    def __init__(self, x, y, w, h):
        """Constructs the necessary attributes for the rect object.

        :param x: Top left corner's x coordinate.
        :param y: Top left corner's y coordinate.
        :param w: Width of the rectangle.
        :param h: Height of the rectangle.
        """
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def __eq__(self, b) -> bool:
        """Compares two rectangles for equality.

        :param b: A rectangle object.
        :return: True if the rectangles are equal, false otherwise.
        """
        return self.x == b.x and self.y == b.y and self.w == b.w and self.h == b.h

    def __str__(self) -> str:
        """Outputs a string with the rectangles attributes.

        :return: String with the attributes of the rectangle.
        """
        return "[x: {0}, y: {1}, w: {2}, h: {3}]".format(self.x, self.y, self.w, self.h)

    def __and__(self, b) -> Rect:
        """Combines two rectangles using the union method.

        :param b: Rectangle to combine.
        :return: A new combined rectangle.
        """
        return self.union(self, b)

    def area(self) -> int:
        """Returns the area of the rectangle."""
        return self.w * self.h

    @staticmethod
    def intersect(a, b) -> bool:
        """Checks if two rectangles intersect.

        :param a: A rectangle to check for intersection.
        :param b: A rectangle to check for intersection.
        :return: True if the rectangles intersect, False otherwise.
        """
        x = max(a.x, b.x)
        y = max(a.y, b.y)
        w = min(a.x + a.w, b.x + b.w) - x
        h = min(a.y + a.h, b.y + b.h) - y
        if h < 0 or w < 0:
            return False
        return True

    @staticmethod
    def union(a, b) -> Rect:
        """Merges two rectangles

        Create a new rectangle the encompasses both rectangles.
        The new rectangle is no bigger than required.
        :param a: A rectangle to merge.
        :param b: A rectangle to merge.
        :return: A new rectangle.
        """
        x = min(a.x, b.x)
        y = min(a.y, b.y)
        w = max(a.x + a.w, b.x + b.w) - x
        h = max(a.y + a.h, b.y + b.h) - y
        return Rect(x, y, w, h)

    @staticmethod
    def overlap(a, b, out_type="percent") -> float:
        """Calculates the overlap of two rectangles

        Can be used to get the percent overlap or the overlap area.
        :param a: A rectangle to calculate the overlap with.
        :param b: A rectangle to calculate the overlap with.
        :param out_type: Determines whether the output is a percentage or an area.
        :return: A float of either the area or percentage overlap.
        """
        x = max(a.x, b.x)
        y = max(a.y, b.y)
        w = min(a.x + a.w, b.x + b.w) - x
        h = min(a.y + a.h, b.y + b.h) - y
        intersection_area = w * h
        if h < 0 or w < 0:
            return 0
        if out_type == "area":
            return intersection_area
        else:
            return intersection_area / (a.area() + b.area() - intersection_area)

    @staticmethod
    def group_rectangles(rectangle_list) -> list:
        """Takes a list of rectangles and merges any that overlap on the list.

        :param rectangle_list: A list of rectangles to group.
        :return: A list of grouped rectangles.
        """
        rects = rectangle_list.copy()
        removal_list = []

        for i in range(len(rects)):
            removal_list.append(False)

        # Rectangles that overlap a merged together.
        # Rectangles that were merged are flagged for removal from the list.
        # Iterate backwards so we can remove while iterating later
        for i in range(len(rects) - 1, -1, -1):
            if removal_list[i]:
                continue
            ra = rects[i]

            for j in range(len(rects) - 1, -1, -1):
                if removal_list[j] or i == j:
                    continue
                rb = rects[j]
                if Rect.intersect(ra, rb):
                    ra = ra & rb  # Union
                    rects[i] = ra
                    removal_list[j] = True

        # Remove all the rectangles that have been merged with another rectangle.
        for i in range(len(rects) - 1, -1, -1):
            if removal_list[i]:
                del rects[i]
        return rects
