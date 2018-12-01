# clip --> labelled_images for frames with faces
# allow * in input clip names

import glob
import os

import cv2
from cam_boxes import BGR_RED, BGR_GREEN
from cam_detect_cfg import cfg
from cam_dnn_pers import PersDetector, FaceDetector  # , FaceRecognizer
from cam_io import InputStream, UserStream
from cam_time_measure import TimeMeasure


def main():
    user_stream = UserStream()
    pers_detector = PersDetector()
    face_detector = FaceDetector()
    tm = TimeMeasure()

    # os.chdir(cfg['util_raw_clips_folder'])

    for label in cfg['face_labels_list']:

        clip_label_folder = cfg['util_raw_clips_folder'] + label + '/'
        img_label_folder = cfg['util_img_folder'] + label + '/'
        print('========================================================================================')
        print('label:{}  clip_label_folder:{}   img_label_folder:{}'.
              format(label, clip_label_folder, img_label_folder))

        os.makedirs(img_label_folder, exist_ok=True)

        for clip_fname in glob.glob('{}*.avi'.format(clip_label_folder)):
            print('----------------------------------------------------------------------')
            print('Processing clip: {}'.format(clip_fname))
            inp_stream = InputStream(input_src=clip_fname)

            while user_stream.wait_key() is True:
                inp_frame = inp_stream.read_frame()
                if inp_frame is None: break

                tm.set('pers detector')

                obj_boxes = pers_detector.detect(inp_frame)  # list of boxes with persons

                tm.set('face detector')

                face_boxes = face_detector.detect(inp_frame)  # ObjBox - list, corners-order

                tm.set('processing results')

                if len(obj_boxes) == 0 or len(face_boxes) == 0: continue

                # remove boxes which are not fitting into limits for w,h
                obj_boxes = [fb for fb in obj_boxes if fb.w_h_is_in_pers_range()]
                if len(obj_boxes) == 0: continue
                face_boxes = [fb for fb in face_boxes if fb.w_h_is_in_face_range()]
                if len(face_boxes) == 0: continue

                # find outer pers box
                pb = None
                for fb in sorted(face_boxes, key=lambda fb: fb.confidence, reverse=True):
                    outer_pb_lst = [pb for pb in obj_boxes if fb.is_inside_box(pb.box)]
                    if len(outer_pb_lst) > 0:
                        pb = outer_pb_lst[0]
                        break
                if pb is None: continue  # skip face if none outer pers box is found

                time_from_clip_fname = clip_fname[len(clip_label_folder) + 15: len(clip_label_folder) + 15 + 19]
                img_fname = '{}_{:4.2f}_{}_{}.png'.format(label, fb.confidence, fb.box.box_2_str(),
                                                          time_from_clip_fname)
                cv2.imwrite(img_label_folder + img_fname, inp_frame)
                print('frame saved to: {}'.format(img_label_folder + img_fname))

                if cfg['lbl_images_boxed_needed']:
                    os.makedirs(cfg['lbl_images_boxed_folder'], exist_ok=True)
                    boxed_images_label_folder = cfg['lbl_images_boxed_folder'] + label + '/'
                    os.makedirs(boxed_images_label_folder, exist_ok=True)

                    if pb is not None:
                        pb.draw(inp_frame, text="conf={:.2f}%".format(pb.confidence * 100), color=BGR_GREEN)
                    fb.draw(inp_frame, color=BGR_RED)

                    cv2.imwrite(boxed_images_label_folder + img_fname, inp_frame)
                    print('boxed image saved to: {}'.format(boxed_images_label_folder + img_fname))

            del inp_stream
    print('===== completed ======')
    print('time measuring:\n', tm.results())


if __name__ == '__main__':
    main()
