"""
cam_dnn_faces.py
include all face-related stuff:  face detection, face recognition (encoding, classifying)
"""

import pickle

import cv2
import face_recognition
import numpy as np

from cam_detect_cfg import cfg
from cam_boxes import ObjBox
from cam_dnn_pers import PersDetector

print('cam_dnn_faces is imported')

class FaceDetector:

    def __init__(self):

        self.net = cv2.dnn.readNetFromCaffe(cfg['face_det_prototxt'], cfg['face_det_model'])
        print(f"Loaded face detection model files: {cfg['face_det_prototxt']} and {cfg['face_det_model']}.")

    def detect(self, frame):

        # grab the frame dimensions and convert it to a blob
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

        # pass the blob through the network and obtain the detections and predictions
        self.net.setInput(blob)
        detections = self.net.forward()

        face_detections = []

        # loop over the detections
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the `confidence` is greater than the minimum confidence
            if confidence < cfg['face_det_confidence']:
                continue

            # compute the (x, y)-coordinates of the bounding box for the object
            idx = detections[0, 0, i, 1]
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])

            (startX, startY, endX, endY) = box.astype("int")

            obj_box = ObjBox(startX, startY, endX, endY, confidence, idx, 'unknown')
            face_detections.append(obj_box)

        return face_detections


# -------------------------------------------------------------------------------------------------------------------

from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, accuracy_score

import warnings

# Suppress LabelEncoder warning
warnings.filterwarnings('ignore')


class FaceRecognizer:

    def __init__(self, method='knn_new'):
        # load the known faces and embeddings
        self.known = pickle.loads(open(cfg['encodings_file'], "rb").read())
        print(f"Loaded {len(self.known['encodings'])} face encodings from {cfg['encodings_file']}")
        assert len(self.known['encodings']) == len(self.known['labels']) == len(self.known['img_fnames']) == len(
            self.known['boxes'])
        self.method = method
        if self.method == 'knn_new':
            self.train_knn()
            self.set_threshold(cfg['face_rec_threshold'])

    def recognize(self, frame, boxes):
        if len(boxes) == 0:
            return []
        if self.method == 'knn_old':
            return self.recognize_knn_old(frame, boxes)
        if self.method == 'knn_new':
            return self.recognize_knn_new(frame, boxes)
        print(f"unknown method {self.method} for FaceRecognizer.recognize")

    def train_knn(self):
        targets = np.array(self.known['labels'])

        self.encoder = LabelEncoder()
        self.encoder.fit(targets)

        # Numerical encoding of identities
        y_train = self.encoder.transform(targets)
        X_train = np.array(self.known['encodings'])

        self.knn = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
        self.knn.fit(X_train, y_train)

    def recognize_knn_new(self, frame, boxes):

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(rgb, boxes)
        predicts = self.knn.predict(encodings)
        labels = self.encoder.inverse_transform(predicts)
        # get distance to nearest neighbor and his (neighbor) index
        distances, indexes = self.knn.kneighbors(encodings, return_distance=True)

        face_boxes = []  # list of ObjBoxes with info about recognized faces

        # loop over the facial embeddings
        assert len(encodings) == len(boxes) == len(labels) == len(distances)
        for box, label, distance, index in zip(boxes, labels, distances, indexes):

            if cfg['face_rec_use_threshold']:
                if distance > self.threshold:
                    label = "unknown"
            obj_box = ObjBox(label=label, confidence=distance[0], idx=index[0], sides_tuple=box)
            face_boxes.append(obj_box)

        return face_boxes

    def set_threshold(self, threshold_val, dataset_folder=None):

        if threshold_val != -1:
            # set threshold by manual value, not auto calculated
            self.threshold = threshold_val
            print(f'face recognition threshold was set to {threshold_val}')
            return

        def distance(emb1, emb2):
            return np.sqrt(np.sum(np.square(emb1 - emb2)))

        if dataset_folder is not None:
            self.test_knn(dataset_folder + 'face_encodings.pkl')
            known = self.test_known
        else:
            known = self.known

        distances = []  # squared L2 distance between pairs
        identical = []  # 1 if same identity, 0 otherwise

        num = len(known['encodings'])

        for i in range(num - 1):
            for j in range(1, num):
                distances.append(distance(known['encodings'][i], known['encodings'][j]))
                identical.append(1 if known['labels'][i] == known['labels'][j] else 0)

        distances = np.array(distances)
        identical = np.array(identical)

        thresholds = np.arange(0.3, 1.0, 0.01)

        f1_scores = [f1_score(identical, distances < t) for t in thresholds]
        # acc_scores = [accuracy_score(identical, distances < t) for t in thresholds]

        opt_idx = np.argmax(f1_scores)
        # Threshold at maximal F1 score
        opt_tau = thresholds[opt_idx]
        # Accuracy at maximal F1 score
        opt_acc = accuracy_score(identical, distances < opt_tau)
        self.threshold = opt_tau

        print(f'Threshold for face recognition = {opt_tau:.2f}. Accuracy at threshold = {opt_acc:.3f}')

    def test_knn(self, encoding_fname):

        # load the known faces and embeddings
        print(f"loading test data encodings from file {encoding_fname}")
        # known = { 'encodings': [n] , 'labels': [n] , 'img_fnames': [n] , 'boxes': [n] }
        self.test_known = pickle.loads(open(encoding_fname, "rb").read())
        assert    len(self.test_known['encodings'])  == len(self.test_known['labels']) \
               == len(self.test_known['img_fnames']) == len(self.test_known['boxes'])

        test_targets = np.array(self.test_known['labels'])

        # Numerical encoding of identities
        y_test = self.encoder.transform(test_targets)
        X_test = np.array(self.test_known['encodings'])

        acc_knn = accuracy_score(y_test, self.knn.predict(X_test))
        print(f'kNN accurary = {acc_knn}')

        return acc_knn

    def recognize_knn_old(self, frame, boxes):

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        encodings = face_recognition.face_encodings(rgb, boxes)
        face_boxes = []  # list of ObjBoxes with info about recognized faces

        # loop over the facial embeddings
        assert len(encodings) == len(boxes)
        for encoding, box in zip(encodings, boxes):
            # attempt to match each face in the input image to our known encodings
            matches = face_recognition.compare_faces(self.known["encodings"], encoding)
            name = "Unknown"

            # check to see if we have found a match
            if True in matches:
                # find the indexes of all matched faces
                matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                # initialize a dictionary to count the total number of times each face was matched
                counts = {}

                # loop over the matched indexes and maintain a count for each recognized face face
                for i in matchedIdxs:
                    name = self.known['labels'][i]
                    counts[name] = counts.get(name, 0) + 1

                # determine the recognized face with the largest number of votes
                name = max(counts, key=counts.get)

            # update the list of names
            obj_box = ObjBox(label=name, sides_tuple=box)
            face_boxes.append(obj_box)

        return face_boxes

    def print_known(self):
        print('Downloaded known encodings:')
        for i in range(len(self.known['encodings'])):
            print('{:4d} label={:15s} fname={:40s} box={}'.format(
                i, self.known['labels'][i], self.known['img_fnames'][i], self.known['boxes'][i]))


# -------------------------------------------------------------------------------------------------------------------

class ImageScanner:
    """
    Hi level wrapper for 'pers_detect -> face_detect -> face_recognize' pipeline.
    """

    def __init__(self, time_meter=None, filter_flg=True, recognizer_method='knn_new'):
        self.pers_detector = PersDetector()
        self.face_detector = FaceDetector()
        self.face_recognizer = FaceRecognizer(recognizer_method)
        self.filter = filter_flg
        self.time_meter = time_meter

    def scan(self, image):

        if self.time_meter is not None:
            self.time_meter.set('pers detect')
        self.pers_boxes = self.pers_detector.detect(image)  # list of boxes with persons

        if self.time_meter is not None:
            self.time_meter.set('face detect')
        self.facedet_boxes = self.face_detector.detect(image)

        if self.time_meter is not None:
            self.time_meter.set('face recognize')
        # convert list of ObxBox to list of sides-orders tuples
        boxes = [fd_box.box.sides() for fd_box in self.facedet_boxes]
        self.face_boxes = self.face_recognizer.recognize(image, boxes)

        if self.time_meter is not None:
            self.time_meter.set('filter')
        if self.filter:
            # remove boxes which are not fitting into limits for w,h
            self.pers_boxes = [fb for fb in self.pers_boxes if fb.w_h_is_in_pers_range()]
            # remove boxes which are not fitting into limits for w,h
            self.face_boxes = [fb for fb in self.face_boxes if fb.w_h_is_in_face_range()]
            # remove faces which are not inside of at least one pers_box
            self.face_boxes = [fb for fb in self.face_boxes if fb.is_inside_obj_boxes(self.pers_boxes)]

        return self.pers_boxes, self.face_boxes


# ------------------ main -------------------------------------------------------------------------------------------


from cam_boxes import BGR_RED
import os
import cam_time_measure
import glob

show_closest_neighbor = True


def main():
    img_folder = 'test_images_folder/Igor/'
    result_folder = img_folder + 'results/'
    os.makedirs(result_folder, exist_ok=True)

    # pers_detector = PersDetector()
    # face_detector = FaceDetector()
    # face_recognizer = FaceRecognizer()
    image_scaner = ImageScanner()

    # face_recognizer.train_knn()
    # face_recognizer.test_knn('test_images_folder/face_encodings.pkl')
    # face_recognizer.print_known()
    # face_recognizer.set_threshold(dataset_folder='test_images_folder/')

    tm = cam_time_measure.TimeMeasure()

    for img_fname in glob.glob(img_folder + '*.png'):

        # print('fname=',img_fname)
        image = cv2.imread(img_fname)

        tm.set('detect & recognize')

        pers_boxes, face_boxes = image_scaner.scan(image)

        # facedet_boxes = face_detector.detect(image)
        # boxes = [fd_box.box.sides() for fd_box in facedet_boxes]
        # face_boxes = face_recognizer.recognize(image, boxes)

        tm.set('write')

        for face_box in face_boxes:
            face_box.draw(image, color=BGR_RED)
            print(f"fname={img_fname:50} label={face_box.label:10} conf={face_box.confidence:4.2f}")
            if show_closest_neighbor:
                known_fnames = image_scaner.face_recognizer.known['img_fnames']
                neighbor_fname = known_fnames[face_box.idx]
                print(f"Closest: {neighbor_fname}")
        cv2.imwrite(result_folder + os.path.split(img_fname)[-1], image)

    print(tm.results())


if __name__ == '__main__':
    main()
