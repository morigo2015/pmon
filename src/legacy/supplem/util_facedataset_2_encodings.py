# USAGE
# python encode_faces.py --dataset dataset --encodings encodings.pickle

# import the necessary packages

import pickle
import cv2
import os

import face_recognition

from cam_detect_cfg import cfg
from cam_time_measure import TimeMeasure
from cam_boxes import Box, ObjBox, BGR_WHITE

tm = TimeMeasure()

# initialize the list of known encodings and known names
known_encodings_info = {'encodings': [], 'labels': [], 'img_fnames': [], 'boxes': []}

for label in cfg['face_labels_list']:
    label_folder = cfg['facedataset_folder'] + label + '/'
    if not os.path.exists(label_folder): continue  # skip labels which have no samples

    for img_fname in os.listdir(label_folder):
        path_to_image = label_folder + img_fname
        print('image file: {}'.format(path_to_image))

        tm.set('prepare')

        # load the input image and convert it from RGB (OpenCV ordering) to dlib ordering (RGB)
        image = cv2.imread(path_to_image)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        tm.set('location')

        # take coordinates from file name
        str_splitted = img_fname.split('_')
        corners_tuple = Box.str_2_coord(str_splitted[2])  # fname: <label>_<conf>_<coordinates>_oldfnametime
        sides_tuple = Box.corners_2_sides(corners_tuple=corners_tuple)
        boxes = [sides_tuple]  # always the only one box from filename

        # items in the boxes[] are side-ordered as face_encoding requires

        for b in boxes:
            print('box:{} w,h:{},{} '.format(b, Box.width(sides_tuple=b), Box.height(sides_tuple=b)))

        tm.set('face_encodings')

        # compute the facial embedding for the face
        encodings = face_recognition.face_encodings(rgb, boxes)

        tm.set('add encodings')

        # loop over the encodings
        for (encoding, box) in zip(encodings, boxes):
            # add each encoding + name to set of known names and encodings

            known_encodings_info['encodings'].append(encoding)
            known_encodings_info['labels'].append(label)
            known_encodings_info['img_fnames'].append(img_fname)
            known_encodings_info['boxes'].append(box)

            print('info: label={}, img_fname={},  box={}'.format(label, img_fname, box))

            if cfg['encoding_boxed_needed']:
                b = ObjBox(sides_tuple=box)
                b.draw(image, color=BGR_WHITE)

                os.makedirs(cfg['encoding_boxed_folder'], exist_ok=True)
                boxed_images_label_folder = cfg['encoding_boxed_folder'] + label + '/'
                os.makedirs(boxed_images_label_folder, exist_ok=True)
                cv2.imwrite(boxed_images_label_folder + img_fname, image)
                print('boxed image saved to {}'.format(boxed_images_label_folder + img_fname))

# dump the facial encodings + names to disk
f = open(cfg['encodings_file'], "wb")
f.write(pickle.dumps(known_encodings_info))
f.close()
print("encodings for {} images saved to {}".format(len(known_encodings_info['encodings']), cfg['encodings_file']))

print('\ntime measuring:\n{}'.format(tm.results()))
