# folders check
# divide files in videos,event_messages,filter_clips depending on whether persons are found by yolo.
# may work in parallel with cam_detect.py

import cv2
import numpy as np
import os.path
import time
import glob

from cam_boxes import Box, ObjBox, BGR_CYAN

input_root = "../cloud/" # "../production/"
folders_to_check = [("videos/",".avi"), ("filtered_clips/",".avi"), ("event_images/",".jpg")]
reverse = True
result_root = input_root + "checked/"
folder_persons = "persons/"
folder_noperson = "noperson/"
log_file = input_root + "filecheck.log"
sleep_seconds = 500

modelConfiguration = "../production/models/yolov3.cfg"
modelWeights = "../production/models/yolov3.weights"
classesFile = "../production/models/coco.names"

# Initialize the parameters
confThreshold = 0.5  #Confidence threshold
nmsThreshold = 0.4   #Non-maximum suppression threshold
inpWidth = 416       #Width of network's input image
inpHeight = 416      #Height of network's input image


def main():
    pdy = PersDetectorYolo()
    out(f'Input root: {input_root}\nFolders to check: {folders_to_check}')
    pers_cnt = [0 for i in folders_to_check]
    nopers_cnt = [0 for i in folders_to_check]
    log = open(log_file,"w+")
    while True:
        for idx,(folder,ext) in enumerate(folders_to_check):
            out(f'Processing folder {folder} ',flush=True)
            os.makedirs(result_root+folder+folder_persons,exist_ok=True)
            os.makedirs(result_root+folder+folder_noperson,exist_ok=True)
            file_lst = sorted(glob.glob(f'{input_root}{folder}*{ext}'), reverse=reverse)
            for file_num,fname in enumerate(file_lst):
                out(f'{file_num:4d}/{len(file_lst):4d} ',end='')
                if pdy.persons_found_in_file(fname):
                    result_folder = folder_persons
                    pers_cnt[idx] += 1
                else:
                    result_folder = folder_noperson
                    nopers_cnt[idx] += 1
                cmd =  f"mv '{fname}' '{result_root}{folder}{result_folder}{os.path.split(fname)[1]}'"
                os.system(cmd)
                #out(cmd)
            print_current_stat(folder,pers_cnt,nopers_cnt)
        out(f'sleeping ...',flush=True)
        time.sleep(sleep_seconds)

def print_current_stat(folder,pers_cnt,nopers_cnt):
    for idx, (folder, ext) in enumerate(folders_to_check):
        out(f'Total for {folder:20}: persons={pers_cnt[idx]:4}  nopers={nopers_cnt[idx]:4}')

def out(msg,end='\n',flush=True):
    print(msg,end=end,flush=flush)
    log=open(log_file,'a')
    log.write(msg)
    if end == '\n':
        log.write('\n')
    log.close()

# -------------------------------------------------------------------------------------------------

class PersDetectorYolo:

    def __init__(self):
        self.net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


    def detect(self, frame):
        # Create a 4D blob from a frame.
        blob = cv2.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)

        # Sets the input to the network
        self.net.setInput(blob)

        # Runs the forward pass to get output of the output layers
        outs = self.net.forward(self.getOutputsNames(self.net))

        # Remove the bounding boxes with low confidence
        pers_boxes = self.postprocess(frame, outs)

        # Put efficiency information. The function getPerfProfile returns
        # the overall time for inference(t) and the timings for each of the layers(in layersTimes)
        t, _ = self.net.getPerfProfile()
        label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
        cv2.putText(frame, label, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
        return pers_boxes

    # Get the names of the output layers
    def getOutputsNames(self, net):
        # Get the names of all the layers in the network
        layersNames = net.getLayerNames()
        # Get the names of the output layers, i.e. the layers with unconnected outputs
        return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]


    # Remove the bounding boxes with low confidence using non-maxima suppression
    def postprocess(self, frame, outs):
        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]

        # Scan through all the bounding boxes output from the network and keep only the
        # ones with high confidence scores. Assign the box's class label as the class with the highest score.
        classIds = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                classId = np.argmax(scores)
                if classId != 0:  # classId for Person ==0, other objects are being skipped
                    continue
                confidence = scores[classId]
                if confidence > confThreshold:
                    center_x = int(detection[0] * frameWidth)
                    center_y = int(detection[1] * frameHeight)
                    width = int(detection[2] * frameWidth)
                    height = int(detection[3] * frameHeight)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    classIds.append(classId)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])

        # Perform non maximum suppression to eliminate redundant overlapping boxes with lower confidences.
        indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
        pers_boxes = []
        for i in indices:
            i = i[0]
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            pers_boxes.append(ObjBox(sides_tuple=(top,left+width,top+height,left), confidence=confidences[i],label='')) # t,r,b,l
        return pers_boxes

    def persons_found_in_file(self, fname):
        out(f'File {fname:100}: ', end='',flush=True)
        ext = fname.split(sep='.')[-1]

        if ext=='avi' or ext=='mp4':
            cap = cv2.VideoCapture(fname)
            pers_found = False
            while True:
                hasFrame, frame = cap.read()
                if not hasFrame:
                    break
                if self.detect(frame):
                    pers_found = True
                    break

        elif ext=='jpg' or ext=='png':
            frame = cv2.imread(fname)
            pers_found = len(self.detect(frame))
        else:
            out(f'unknown file extention {ext}')
            return None

        out(f'{"PERS" if pers_found else "NOPER"}')
        return pers_found


if __name__ == '__main__':
    main()