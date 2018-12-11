""" camera.py - actions with video cameras"""

import csv
import time
import logging
from typing import List

import cv2
import numpy as np

from config import cfg


class Camera:
    cam_list: List["Camera"] = []
    cam_list_size: int = 0

    def __init__(self, cam_name: str, access_str: str):
        self.cam_id: int = Camera.cam_list_size
        Camera.cam_list_size += 1
        self.cam_name: str = cam_name
        self.read_ok: bool = False
        self.image: np.ndarray = None
        self._access_str: str = access_str
        self._handle: cv2.VideoCapture = None
        self.reopen_attempts_cnt: int = 0 # attempts to reopen after error

    def __str__(self) -> str:
        return f"Cam({self.cam_id},{self.cam_name})"

    def __repr__(self) -> str:
        return self.__str__()

    def open(self) -> bool:
        self._handle = cv2.VideoCapture(self._access_str)
        return self._handle.isOpened()

    def get_frame(self) -> bool:
        """ read next frame from cam, return True if OK"""
        ret, self.image = self._handle.read()
        self.read_ok = False if ret is False or self._handle.isOpened is False else True
        if self.read_ok:
            self.reopen_attempts_cnt = 0 # cam is ok
        return self.read_ok

    def close(self):
        self._handle.release()
        self._handle = None

    def reopen(self,timeout: int =5):
        """ try to reopen video stream after error """
        self.close()
        time.sleep(timeout)
        self.reopen_attempts_cnt += 1
        self.open()
        logging.debug(f'Tried to reopen {self}. Attempts counter={self.reopen_attempts_cnt}. '
                      f'isOpened={self._handle.isOpened()}')
        return self._handle.isOpened()

    @classmethod
    def init_cameras(cls):
        """ init list of cameras based on file-stored camera info """
        with open(cfg['camera_info_file'], newline='') as csvfile:
            reader = csv.reader(csvfile)
            _ = reader.__next__()  # skip headers
            for row in reader:
                if row[0].strip() == '+':
                    cls.cam_list.append(Camera(cam_name=row[1], access_str=row[2]))

    @classmethod
    def print_cameras(cls):
        print('Camera list: ')
        for cam in cls.cam_list:
            print(cam)

    @classmethod
    def check_cameras(cls):
        for cam in cls.cam_list:
            cam.open()
            print(f"{cam} - {'OK' if cam._handle.isOpened() else 'BAD'}")
            cam.close()


if __name__ == "__main__":
    Camera.init_cameras()
    Camera.print_cameras()
    # Camera.check_cameras()
