import os
import glob
import shutil
import cv2
import time
import numpy as np


from time_measure import  TimeMeasure

folder = 'event_images/'
tmp_folder = 'temp/'

for i in range(10):
    print(f'i={i}')
    os.makedirs(tmp_folder, exist_ok=True)
    cnt=0
    img = cv2.imread('tst/test_io.png')
    tm = TimeMeasure()

    for cnt in range(1500):
        #if cnt % 100 ==0: print(f"cnt={cnt}")
        tm.set('write')
        cv2.imwrite(tmp_folder+f'test_{cnt}.png',img)

        #np.save(f'{tmp_folder}test_{cnt}.png',img)
        tm.set('prepare')
        w=cnt%1200
        h=cnt%700
        img[h][w]=(0.,0.,0.)

    print(tm.results())
    del tm
    print('-----------------------------------------------------------------------')

    shutil.rmtree(tmp_folder)

