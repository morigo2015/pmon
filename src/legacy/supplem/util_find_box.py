
import cv2
import numpy as np
import os
import pickle

from cam_boxes import Box, BoxesArray, BGR_RED, BGR_GREEN

def fname_2_box(fname):
    """
    return box object based on fname, or None if not found
    box corners supposed to be included in fname as substr '(<4dig><4dig><4dig><4dig>)'
    """
    start_pos = fname.find('(')
    end_pos = fname.find(')')
    if start_pos == -1 or end_pos == -1 or (start_pos + 16 + 1 != end_pos):
        print(f'cannot find box info in file name {fname}')
        return None
    corn_tuple = Box.str_2_coord(fname[start_pos + 1:end_pos])
    sides_tuple = Box.corners_2_sides(corners_tuple=corn_tuple)
    return Box(sides_tuple=sides_tuple)

if __name__ == "__main__":

    import glob

    # inp_dir = '/home/im/mypy/cam_detect/preparation/fake_spots/doorbell/'
    inp_dir = '/home/im/mypy/cam_detect/preparation/fake_spots/'
    box_dir = inp_dir+'boxed/'
    os.makedirs(box_dir,exist_ok=True)
    #file_list = [f'{inp_dir}test.jpg']
    file_list = glob.glob(f'{inp_dir}*.jpg')

    box_arr = BoxesArray()
    for fname in file_list:
        f = cv2.imread(fname)
        box = fname_2_box(fname)
        if box is None: continue

        box.draw(f, BGR_RED)
        print(f'fname={fname}, count={box_arr.counter} box={box.corners()}  search={box_arr.search(box,0)}')

        box_arr.add_box(box)

    frame = np.ones((720,1280,3),dtype='uint8')
    frame = box_arr.draw(frame)
    cv2.imwrite(box_dir+'boxes.png',frame)

    pickle_fname = box_dir+'fake_spots.pkl'
    box_arr.save_to_disk(pickle_fname)


    # test of save/load
    box_arr2 = BoxesArray(pickle_fname)
    print(f'loaded. counter={box_arr2.counter} ')

    r1 = box_arr.get_box(100)
    r = box_arr2.search(r1,4)
    print(f'Check of save/load. r1={r1.corners()},r={r}')
