import threading
import datetime
import time
import logging
import os
from queue import Queue
from typing import Dict, List, Tuple

import numpy as np
import cv2

from config import cfg
from camera import Camera

_frames_queue = Queue(cfg['camera_frames_que_size'])  # put: frames from CamWorker threads; get: FrameProcessor
_stop_flag = False  # set True to stop all threads


class Frame:
    """ camera frame (image, cam info, timestamp) """

    def __init__(self, cam_id: int, image: np.ndarray, timestamp: str):
        self.cam_id: int = cam_id
        self.cam_name: str = Camera.cam_list[cam_id].cam_name
        self.timestamp: str = timestamp
        self.image: np.ndarray = image

    def __str__(self):
        return f"Frame({self.cam_id},{self.cam_name},{self.timestamp})"


class _CamWorker(threading.Thread):
    """ one thread per each cam """

    def __init__(self, cam: Camera):
        super().__init__(name="_worker_" + cam.cam_name)
        self.cam: Camera = cam
        self.frames_cnt = 0  # frames received from cam (for monitoring)

    def run(self):
        """ run endless: take image, create frame, put in the queue """
        _CamMonitor.msg(self.cam, f"waiting for connect ...")
        self.cam.open()
        _CamMonitor.msg(self.cam, "connected")
        logging.info(f"{self.cam} connected")
        while not _stop_flag:
            if not _frames_queue.full():
                self.cam.get_frame()
                if self.cam.read_ok:
                    frame = Frame(self.cam.cam_id, self.cam.image,
                                  datetime.datetime.now().strftime("%y-%m-%d_%H:%M:%S:%f"))
                    _frames_queue.put(frame)
                    self.frames_cnt += 1
                    if self.frames_cnt % cfg['cam_monitor_frame_div'] == 0:
                        _CamMonitor.msg(self.cam, f"frames processed: {self.frames_cnt}")
                    logging.debug(f'Put {frame}. Quesize={_frames_queue.qsize()}')
                else:  # read error
                    logging.warning(f"Read error in {self.cam}")
                    _CamMonitor.msg(self.cam, f"Error. Failed attempts to reopen: {self.cam.reopen_attempts_cnt}")
                    self.cam.reopen()
                    continue


class _FrameProcessor(threading.Thread):
    """ process main queue: take frames, call appropriate handlers """

    def __init__(self):
        super().__init__(name='FrameProcessor')
        return

    def run(self):
        while not _stop_flag:
            if not _frames_queue.empty():
                frame: Frame = _frames_queue.get()
                logging.debug(f'Get {frame}. Qsize = {_frames_queue.qsize()}')
                if cfg['show_frames']:
                    _show_frame(frame)
                if cfg['write_frames']:
                    _FrameWriter.write_frame(frame)


class _FrameWriter:
    """ Write frames to videofile

    videofiles are being created when first frame is arrived (to avoid manual configuring of shape info)
    """
    _handles: Dict[int, cv2.VideoWriter] = {}

    @classmethod
    def _create_video_file(cls, frame: Frame) -> cv2.VideoWriter:
        if frame.cam_id in cls._handles.keys():  # is already opened
            return cls._handles[frame.cam_id]
        folder = cfg['write_frames_folder']
        os.makedirs(folder, exist_ok=True)
        file_name = f"{folder}{frame.cam_name}{cfg['write_frames_suffix']}"
        shape = (frame.image.shape[1], frame.image.shape[0])
        fh = cv2.VideoWriter(file_name, cfg['write_frames_four_cc'], cfg['write_frames_fps'], shape)
        cls._handles[frame.cam_id] = fh
        logging.debug(f"Video file {file_name} created: shape={shape}, fps={cfg['write_frames_fps']}")
        return fh

    @classmethod
    def write_frame(cls, frame: Frame):
        try:
            fh = cls._handles[frame.cam_id]
        except KeyError:
            fh = cls._create_video_file(frame)
        fh.write(frame.image)

    @classmethod
    def close_all(cls):
        """ close all opened video files """
        for cam_id in cls._handles.keys():
            cls._handles[cam_id].release()
            logging.debug(f'File for {Camera.cam_list[cam_id]} is closed')


class _CamMonitor:
    """ display current state and activity of the cams """

    x0, y0 = None, None  # overall message area left-up corner
    x_p, y_p = None, None  # cursor parking point outside overall message area

    @classmethod
    def init(cls, left_top_corner_xy=(20, 1)):
        if not cfg['cam_monitor']:
            return
        cls.clear_screen()
        cls.x0, cls.y0 = left_top_corner_xy[0], left_top_corner_xy[1]
        cls.x_p, cls.y_p = left_top_corner_xy[0], cls.y0 + 2 + len(Camera.cam_list)

    @classmethod
    def msg(cls, cam: Camera, message: str):
        if not cfg['cam_monitor']:
            return
        # print message:
        y = cls.y0 + cam.cam_id
        cls.set_cursor_to(cls.x0, y)
        print(f"{f'{cam}':.<20s}: {message:<45s}")
        # park cursor
        cls.set_cursor_to(cls.x_p, cls.y_p)

    @staticmethod
    def set_cursor_to(x, y):
        print(f"\033[s\033[{y};{x}f", end='')

    @staticmethod
    def clear_screen():
        print(f"\033[2J", end='')


# vserv own functions:

def _show_frame(frame: Frame):
    """ show cam frame in opencv window related to this cam """
    cv2.imshow(frame.cam_name, frame.image)
    ch = cv2.waitKey(1) & 0xFF
    if ch in [ord('q'), ord('Q'), 27]:
        logging.warning('Cancelled by user')
        stop_vserv()


def _wait_workers_to_stop():
    logging.debug("waiting cam workers to stop:")
    while True:
        time.sleep(1)
        workers_left = len([t.name for t in threading.enumerate() if t.name.startswith("_worker_")])
        logging.debug(f"{workers_left} workers left ...")
        if not workers_left:
            break
    logging.debug("all workers stopped")


def _set_server_logging():
    log_level = getattr(logging, cfg['log_level'].upper(), None)
    # logging.basicConfig(level=log_level, style='{', format='{threadName:22s}:{message}')
    log_config = {
        'level': log_level,
        'style': '{',
        'format': '{threadName:22s}:{message}'
    }
    if cfg['log_file']:
        log_config['filename'] = cfg['log_file']
        log_config['filemode'] = cfg['log_file_mode']
    logging.basicConfig(**log_config)
    logging.debug(f"logging init: level={log_level} log_file={cfg['log_file']}")


def stop_vserv():
    """ Stop work. Close all threads"""
    global _stop_flag
    _stop_flag = True
    _FrameWriter.close_all()  # close opened write-streams
    cv2.destroyAllWindows()
    _wait_workers_to_stop()
    logging.debug('Video Server finished')


def run_server():
    _set_server_logging()
    Camera.init_cameras()
    _CamMonitor.init((1, 1))

    fp = _FrameProcessor()
    fp.start()
    logging.debug('FrameProcessor started')

    for cam in Camera.cam_list:
        _CamWorker(cam).start()
        logging.debug(f'CamWorker for {cam} started')

    try:
        fp.join()  # just wait till FrameProcessor will be stopped
    except KeyboardInterrupt:
        logging.debug('Keyboard interrupt')
        stop_vserv()


def _main():
    run_server()


if __name__ == "__main__":
    _main()
