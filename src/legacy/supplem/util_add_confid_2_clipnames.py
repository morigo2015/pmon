# add facedet confidence to the name of clip (to sort by conf)
# allow * in input clip names

import glob
import os

from cam_detect_cfg import cfg
from cam_dnn_pers import FaceDetector  # , FaceRecognizer
from cam_io import InputStream, UserStream


def main():
    user_stream = UserStream()
    face_detector = FaceDetector()

    os.chdir(cfg['util_clip_folder'])

    for clip_fname in glob.glob(cfg['util_clip_input']):

        print('Processing clip: {}'.format(clip_fname))
        inp_stream = InputStream(input_src=clip_fname)

        max_confidence = 0.0

        while user_stream.wait_key() is True:
            inp_frame = inp_stream.read_frame()
            if inp_frame is None: break

            face_boxes = face_detector.detect(inp_frame)

            for fb in face_boxes:
                confidence = fb.confidence
                max_confidence = max(max_confidence, confidence)

        del inp_stream

        new_fname = '_{:4f}_{}'.format(max_confidence, clip_fname)
        os.rename(clip_fname, new_fname)
        print('Rename {} ----> {}'.format(clip_fname, new_fname))


if __name__ == '__main__':
    main()
