# test recognizer
# get images from test_folder, analyze face inside, write to analyze-labelled folder, collect stat
# aims:
# 1) compare different recognizers
# 2) find labels which are hard for current recognizer

import os

import cv2
from cam_boxes import BGR_RED
from cam_detect_cfg import cfg
from cam_dnn_pers import ImageScanner
from cam_time_measure import TimeMeasure


# cfg['face_labels_list'] = ['Olka']

def main():
    tm = TimeMeasure()
    image_scanner = ImageScanner(time_meter=tm, filter_flg=True, recognizer_method='knn_new')

    test_folder = 'face_dataset/'  # cfg['test_images_folder']
    result_prefix = "_analyzed/"
    result_cnt = {}
    result_opts = ['not_found', 'multifaces', 'ok', 'bad', 'bad/total']

    for label in cfg['face_labels_list']:
        label_folder = test_folder + label + '/'
        result_cnt[label] = {}
        for r in result_opts:
            result_cnt[label][r] = 0
        if not os.path.exists(label_folder):
            print(f"no samples for label '{label}'")
            continue  # skip labels which have no samples

        for img_fname in os.listdir(label_folder):
            path_to_image = label_folder + img_fname

            image = cv2.imread(path_to_image)

            pers_boxes, face_boxes = image_scanner.scan(image)

            if len(face_boxes) == 0:
                result = 'not_found'
                result_folder = test_folder + result_prefix + "_not_found/"

            else:  # len(facedet_boxes) 1 or more:
                for face_box in face_boxes:
                    face_box.draw(image, color=BGR_RED)

                if len(face_boxes) == 1:
                    face_box = face_boxes[0]

                    file_label = img_fname.split(sep='_')[0]
                    if file_label == face_box.label:
                        result = 'ok'
                    else:
                        result = 'bad'
                    result_folder = test_folder + result_prefix + face_box.label + '/'

                else:  # len(face_boxes) > 1
                    result = 'multifaces'
                    result_folder = test_folder + result_prefix + "_multifaces/"

            os.makedirs(result_folder, exist_ok=True)
            cv2.imwrite(result_folder + img_fname, image)
            print(f"result={result:12}   {img_fname}: {label_folder} ---> {result_folder}")
            result_cnt[label][result] += 1

    print('\n\n', tm.results())

    total = print_double_dictionary(result_cnt, result_opts)

    print(f"\n{' ':12}bad/(ok+bad)")
    for label in cfg['face_labels_list']:
        bt_ratio = result_cnt[label]['bad'] / float(result_cnt[label]['bad'] + result_cnt[label]['ok'])
        print(f"{label:>12}: {bt_ratio*100:>6.2f}%")

    bt_ratio = total['bad'] / float(total['bad'] + total['ok'])
    print(f"{'Total':12}: {bt_ratio*100:>6.2f}%")


def print_double_dictionary(dic, opts):
    """
    print table which is double dictionary: each row has unique label and columns opt1, opt2, ..., optn
    :param dic:
    :param opts: list of columns name (not include first column which is 'label' always)
    :return:
    """

    total = {}
    for r in opts + ['sum']: total[r] = 0

    # 1. Print header

    print(f"{'':12s} ", end='')
    for r in opts + ['sum']:
        print(f"{r:>11s} ", end='')
    print(' ')

    # 2. Print body

    for lbl in cfg['face_labels_list']:
        print(f"{lbl:12s} ", end='')
        for r in opts:
            print(f"{dic[lbl][r]:11} ", end='')
        dic[lbl]['sum'] = sum([dic[lbl][r] for r in opts])
        print(f"{dic[lbl]['sum']:11}")

    # 3. Print footer
    # calc totals
    for lbl in cfg['face_labels_list']:
        for r in opts + ['sum']:
            total[r] += dic[lbl][r]
    # print totals
    print(f"{'Total:':12s} ", end='')
    for r in opts + ['sum']:
        print(f"{total[r]:11} ", end='')
    print(' ')

    return total


if __name__ == '__main__':
    main()
