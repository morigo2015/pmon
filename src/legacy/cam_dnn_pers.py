# dnn related stuff (detection, recognition)

import cv2
import numpy as np
import datetime
import os

from cam_boxes import ObjBox, Box, BoxesArray, BGR_RED, BGR_GREEN
from cam_detect_cfg import cfg
from cam_time_measure import TimeMeasure

class PersDetector:
    # class labels MobileNet SSD was trained to detect
    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
               "sofa", "train", "tvmonitor"]

    def __init__(self):
        self.net = cv2.dnn.readNetFromCaffe(cfg['pers_det_prototxt'], cfg['pers_det_model'])
        print(f"Loaded person detection model files:\n     proto: {cfg['pers_det_prototxt']},\n     model: {cfg['pers_det_model']}.")
        self.fake_spots = BoxesArray(cfg['pers_fake_spots_file'])

    def detect(self, frame):

        TimeMeasure.set('  persdet-blobFromImage')
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)

        TimeMeasure.set('  persdet-forward')
        # pass the blob through the network and obtain the detections and predictions
        self.net.setInput(blob)
        detections = self.net.forward()

        TimeMeasure.set('  persdet-processing')
        obj_boxes = []
        # loop over the detections
        for i in np.arange(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the `confidence` is greater than the minimum confidence
            if confidence < cfg['pers_det_confidence']:
                continue

            # extract the index of the class label from the `detections`,
            # then compute the (x, y)-coordinates of the bounding box for the object
            idx = int(detections[0, 0, i, 1])
            label = PersDetector.CLASSES[idx]
            if label != 'person':
                continue

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # if cfg['pers_fake_spots_filter'] and self.box_is_fake_spot(box,frame):
            #     continue # skip box if it's a fake spot

            obj_box = ObjBox(startX, startY, endX, endY, confidence, idx, label)
            obj_boxes.append(obj_box)

        return obj_boxes

    # def box_is_fake_spot(self, box,frame):
    #     """
    #     check if box is close to one of fake_spots
    #     important: box is ndarray here, not item of Box class
    #     """
    #     box = Box(corners_tuple= box.astype("int"))
    #     if self.fake_spots.search(box, cfg['pers_fake_spots_eps']) == -1:
    #         return False # box isn't fake spot
    #     else:
    #         if cfg['pers_filtered_save']:
    #             self.save_fake_spot(frame, box)
    #         return True # is fake spot
    #
    # def save_fake_spot(self, frame, box):
    #     os.makedirs(cfg['pers_filtered_save_folder'],exist_ok=True)
    #     timestamp = datetime.datetime.now().isoformat(timespec='seconds')
    #     box_str = box.box_2_str()  # corners-order
    #     fname = f"{cfg['pers_filtered_save_folder']}{timestamp}({box_str}){cfg['image_ext']}"
    #     if cfg['pers_filtered_save_boxed']:
    #         box.draw(frame)
    #     cv2.imwrite(fname, frame)
    #     print(f'fake spot filtered. Frame ')
    #     return fname

# -------------------------------------------------------------------------

if __name__ == "__main__":

    import glob
    import os
    from cam_boxes import Box, BGR_RED, BGR_GREEN

    # inp_dir = '/home/im/mypy/cam_detect/preparation/fake_spots/doorbell/'
    inp_dir = '/home/im/mypy/cam_detect/tst/'
    box_dir = inp_dir+'boxed/'
    os.makedirs(box_dir,exist_ok=True)

    pers_detector = PersDetector()

    for fname in glob.glob(f'{inp_dir}*.png'):
        inp_frame = cv2.imread(fname)
        pers_boxes = pers_detector.detect(inp_frame)  # list of boxes with persons
        for pers_box in pers_boxes:
            pers_box.draw(inp_frame, color=BGR_RED)
        cv2.imwrite(f'{box_dir}{fname.split(sep="/")[-1]}',inp_frame)
        print(f'fname={fname},  boxes={[ b.box.sides() for b in pers_boxes  ]}')

