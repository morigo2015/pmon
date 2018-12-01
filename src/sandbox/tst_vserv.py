# tst vserv
# read frames, save to dbserver
#

src = 'rtsp://admin:F112358f_bullet@192.168.1.70:554/Streaming/Channels/101'

import cv2
import mysql.connector
import numpy as np
from my_db import MyDb

db=MyDb()
db.exec('drop table frames')
db.exec('CREATE TABLE IF NOT EXISTS frames (id int, frame LONGBLOB )')

# img=cv2.imread('/home/im/mypy/vint/images/tst/frames.png')
#
# db.save_img(img,id=1)
# rest_img = db.load_img(id=1)
# cv2.imwrite('/home/im/mypy/vint/images/tst/frames-restored.png',rest_img)


cap = cv2.VideoCapture(src)
frame_cnt = 0
while True:
    ret, frame = cap.read()
    if ret == False:
        break
    frame_cnt += 1
    db.save_img(frame,frame_cnt)


