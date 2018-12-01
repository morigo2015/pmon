""" Box, PersBox classes """

import logging
from typing import Tuple, List

import numpy as np


class Box:
    """ rectangle box """

    def __init__(self, ul: Tuple[int, int], br: Tuple[int, int]):
        self.x_ul: int = ul[0]  # upper-left x
        self.y_ul: int = ul[1]  # upper-left y
        self.x_br: int = br[0]  # bottom-right x
        self.y_br: int = br[1]  # bottom-right y

    def __str__(self):
        return f"Box({self.x_ul},{self.y_ul},{self.x_br},{self.y_br})"

    def __repr__(self):
        return self.__str__()

    def draw(self, image: np.ndarray):
        pass


class PersBox(Box):
    """ Person box - one person box detected in image """

    def __init__(self, box: Box, confidence: float, roi: int):
        super().__init__((box.x_ul, box.y_ul), (box.x_br, box.y_br))
        self.confidence: float = confidence
        self.roi: int = roi
        pass

    def __str__(self):
        return f"PBox({self.x_ul},{self.y_ul},{self.x_br},{self.y_br};{self.roi};{self.confidence})"

    def __repr__(self):
        return self.__str__()

    def draw(self, image: np.ndarray):
        pass


class PersBoxLst:
    """ List[PersBox] """

    def __init__(self):
        self.lst: List[PersBox] = []

    def __str__(self):
        return f"PBox[{[pb for pb in self.lst]}]"

    def __repr__(self):
        return self.__str__()

    def append(self, item):
        self.lst.append(item)


if __name__ == "__main__":
    b1 = Box((0,0),(100,100))
    b2 = Box((1,1),(5,5))
    print(f"b1={b1} b2={b2}")

    pb1 = PersBox(b1,0.0,0)
    print(pb1)


