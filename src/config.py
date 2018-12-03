# config.py

import os
import cv2

work_dir = os.environ['HOME'] + '/' + 'mypy/pmon/'
os.chdir(work_dir)

_folders = {
    'src': work_dir + 'src/',
    'data': work_dir + 'data/',
    'videos': work_dir + 'videos/',
    'logs': work_dir + 'logs/',
}

cfg = {
    #
    'show_frames': False,  # display input frames from cameras

    # logging:
    'log_level': 'debug',  # debug/info/warning/error
    'log_file': _folders['logs'] + 'log.txt',  # set '' to send to standard output
    'log_file_mode': 'w',  # 'a' or 'w'

    # cam monitoring:
    'cam_monitor': True,
    'cam_monitor_frame_div': 10, # how many frames between updates of cam status on monitor

    # write frames:
    'write_frames': False,  # write input frames to video files (separated by cam)
    'write_frames_folder': _folders['videos'],
    'write_frames_fps': 3.0,  # fps for videofiles that save cam frame stream
    'write_frames_four_cc': cv2.VideoWriter_fourcc(*'XVID'),  # four_cc for videofiles
    'write_frames_suffix': '.avi',

    # others:
    'camera_info_file': _folders['data'] + 'cameras_info.csv',
    'camera_frames_que_size': 10,  # frames queue size for each cam
}
