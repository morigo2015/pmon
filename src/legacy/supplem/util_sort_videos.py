# removed files from videos to arch_videos while separated by folders

import os

from cam_detect_cfg import cfg


os.chdir('/home/im/mypy/cam_detect/')
inp = 'production/videos/'
out = 'preparation/arch_video/'

for lbl in cfg['face_labels_list']+['faced','boxed','no_faces','clip_']:
    os.makedirs(f'{out}{lbl}/',exist_ok=True)
    cmd_str = f'mv   {inp}*{lbl}*   {out}{lbl}/'
    print('cmd_str=',cmd_str)
    os.system( cmd_str )

