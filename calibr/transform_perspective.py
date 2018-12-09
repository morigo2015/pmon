import json
import numpy as np
import cv2

from colors import BGR_BLACK


class TransformPerspective:
    calibr_json = '[[0.35170874760003623, -0.120010372485569, 86.41017013782465], ' \
                  '[-0.07921794433374281, 0.2704498997571617, 101.44805742481603], ' \
                  '[6.897423792289868e-06, -0.00066381978017083, 1.0]]'
    calibr_wh = (847, 567)

    def __init__(self, shape=(720, 1280), resize_to_shape=True):
        self.trans_mat = np.array(json.loads(TransformPerspective.calibr_json))
        self.calibr_wh = TransformPerspective.calibr_wh
        self.shape = shape[:2]
        self.resize_to_shape = resize_to_shape
        if resize_to_shape:
            ratio_w = self.shape[1] / TransformPerspective.calibr_wh[0]  # shape: (h,w) but calibr_wh: (w,h)
            ratio_h = self.shape[0] / TransformPerspective.calibr_wh[1]  # shape: (h,w) but calibr_wh: (w,h)
            self.resize_ratio = min(ratio_w, ratio_h)  # we have to save w-h proportion so the same ratio for w and h

            delta_w = int(shape[1] - TransformPerspective.calibr_wh[0] * self.resize_ratio)
            delta_h = int(shape[0] - TransformPerspective.calibr_wh[1] * self.resize_ratio)
            self.top, self.bottom = delta_h // 2, delta_h - (delta_h // 2)
            self.left, self.right = delta_w // 2, delta_w - (delta_w // 2)

    def transform_image(self, orig_img):
        trans_img = cv2.warpPerspective(orig_img, self.trans_mat, self.calibr_wh,
                                        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
        if self.resize_to_shape:
            trans_img = cv2.resize(trans_img, None,
                                   fx=self.resize_ratio, fy=self.resize_ratio, interpolation=cv2.INTER_AREA)
            trans_img = cv2.copyMakeBorder(trans_img,
                                           self.top, self.bottom, self.left, self.right,
                                           cv2.BORDER_CONSTANT, value=BGR_BLACK)
        return trans_img


if __name__ == "__main__":
    img = cv2.imread("/home/im/mypy/vint/images/input/grid.png")
    tp = TransformPerspective(img.shape)
    print(f"resize_ratio={tp.resize_ratio:.2f} calibr_wh={tp.calibr_wh} shape={tp.shape}")
    trans_img = tp.transform_image(img)
    cv2.imwrite("/home/im/mypy/vint/images/tmp/grid-transformed.png", trans_img)
