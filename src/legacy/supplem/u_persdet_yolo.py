# read video file, find persons show and write results to another file
import cv2
import argparse
import sys
import numpy as np
import os.path

from cam_boxes import Box, ObjBox, BGR_CYAN

modelConfiguration = "../production/models/yolov3.cfg"
modelWeights = "../production/models/yolov3.weights"
classesFile = "../production/models/coco.names"

outputFile = "../production/yolo_out_py.avi"
src = "/home/im/mypy/cam_detect/cloud/checked/filtered_clips/persons/bell_2018-09-25T22:45:19_099_1_2_0000001012790719_0000000512790719.avi"

# Initialize the parameters
confThreshold = 0.5  #Confidence threshold
nmsThreshold = 0.4   #Non-maximum suppression threshold
inpWidth = 416       #Width of network's input image
inpHeight = 416      #Height of network's input image

def main():
    pdy = PersDetectorYolo()

    cv2.namedWindow('yolo', cv2.WINDOW_NORMAL)
    cap = cv2.VideoCapture(src)
    vid_writer = cv2.VideoWriter(outputFile, cv2.VideoWriter_fourcc('M','J','P','G'), 30,
                                 (round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    while cv2.waitKeyEx(1) != 27:

        # get frame from the video
        hasFrame, frame = cap.read()

        # Stop the program if reached end of video
        if not hasFrame:
            print("Finish! Output file is stored as ", outputFile)
            break

        pers_boxes = pdy.detect(frame)
        for pb in pers_boxes:
            pb.draw(frame, color=BGR_CYAN)
        # cv2.imwrite(outputFile, frame.astype(np.uint8));
        vid_writer.write(frame.astype(np.uint8))

        cv2.imshow('yolo', frame)

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

if __name__ == '__main__':
    main()