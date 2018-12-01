# Boxes

import cv2
import pickle
import numpy as np

from cam_detect_cfg import cfg
from colors import *

class Box:

    # colour = BGR_GREEN  # default colour for box (green)

    def __init__(self, startX=None, startY=None, endX=None, endY=None, coordinate_str=None
                 , sides_tuple=None, corners_tuple=None, box=None):
        """
        create new box based on corners coordinates or corner-order string
        :param coordinate_str: corner-order string
        """
        if coordinate_str is not None:  # init from string, usually - part of file name
            self.startX, self.startY, self.endX, self.endY = Box.str_2_coord(coordinate_str)
        elif startX is not None and startY is not None and endX is not None and endY is not None:
            self.startX = startX
            self.startY = startY
            self.endX = endX
            self.endY = endY
        elif sides_tuple is not None:
            self.startX, self.startY, self.endX, self.endY = Box.sides_2_corners(sides_tuple=sides_tuple)
        elif corners_tuple is not None:
            self.startX, self.startY, self.endX, self.endY = corners_tuple
        elif box is not None:
            self.startX = box.startX
            self.startY = box.startY
            self.endX = box.endX
            self.endY = box.endY
            self.is_empty = box.is_empty
        else:
            print('!!!!! error while initializing Box !!!!!! ')
        self.is_empty = False

    def repr(self):
        return f'{"empty" if self.is_empty else ""}box<({self.startX},{self.startY}),({self.endX},{self.endY})>'

    def _check_limits(self,frame):
        def _limit(val, max_val):
            if val <0: return 0
            if val > max_val-1: return max_val-1
            else: return val
        # check if coordinates are inside a frame limit
        self.startX = _limit(self.startX,frame.shape[1])
        self.startY = _limit(self.startY,frame.shape[0])
        self.endX = _limit(self.endX,frame.shape[1])
        self.endY = _limit(self.endY,frame.shape[0])

    def draw(self, frame, color=None, label=None, thickness=1):
        self._check_limits(frame)
        if color is None:
            color = BGR_GREEN
        cv2.rectangle(frame, (self.startX, self.startY), (self.endX, self.endY), color, thickness) # 1 <--> 2
        if label is not None:
            y = self.startY - 15 if self.startY - 15 > 15 else self.startY + 15
            cv2.putText(frame, label, (self.startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

    def draw2(self,frame,color=None, label=None, thickness=1):
        if self.is_empty:
            return
        else:
            self.draw(frame,color,label, thickness)

    def center(self):
        x_center = self.startX + int((self.endX-self.startX)/2)
        y_center = self.startY + int((self.endY-self.startY)/2)
        return ( x_center, y_center )

    def draw_center(self, frame, color=None, size=None):
        self._check_limits(frame)
        if color is None:
            color = BGR_GREEN
        if size is None:
            size = 5
        x,y = self.center()
        cv2.circle(frame,(x,y), radius=size, color=color, thickness= -1 )

    def sides(self):
        # return top,right,bottom,left
        return self.startY, self.endX, self.endY, self.startX

    def corners(self):
        return self.startX, self.startY, self.endX, self.endY

    @staticmethod
    def width(corners_tuple=None, sides_tuple=None):
        if sides_tuple is not None:
            top, right, bottom, left = sides_tuple
            return right - left
        return

    @staticmethod
    def height(corners_tuple=None, sides_tuple=None):
        if sides_tuple is not None:
            top, right, bottom, left = sides_tuple
            return bottom - top
        return

    def intersect(self,box):
        if self.is_empty or box.is_empty:
            self.is_empty = True
            return self
        self.startX = max(self.startX,box.startX)
        self.startY = max(self.startY,box.startY)
        self.endX = min(self.endX,box.endX)
        self.endY = min(self.endY,box.endY)
        if self.startX >= self.endX or self.startY >= self.endY:
            self.is_empty = True
        return self

    def union(self,box):
        if self.is_empty: return box
        if box.is_empty: return self
        self.startX = min(self.startX,box.startX)
        self.startY = min(self.startY,box.startY)
        self.endX = max(self.endX,box.endX)
        self.endY = max(self.endY,box.endY)
        if self.startX >= self.endX or self.startY >= self.endY:
            self.is_empty = True
        return self

    def area(self):
        if self.is_empty:
            return 0
        else:
            return (self.endX-self.startX)*(self.endY-self.startY)

    def __eq__(self, other):
        if isinstance(other, Box):
            return self.startX == other.startX and self.startY == other.startY \
                   and self.endX == other.endX and self.endY == other.endY
        return NotImplemented

    def box_2_str(self):
        """
        coordinates of existing box --> string for filename (in corners-order)
        return 'corners'-order(sXYeXY), not 'sides'-order(trbl) !!
        """
        return '{:04d}{:04d}{:04d}{:04d}'.format(self.startX, self.startY, self.endX, self.endY)

    @staticmethod
    def str_2_coord(str):
        """
        string for filename -->  box coordinates
        """
        startX = int(str[0:4])
        startY = int(str[4:8])
        endX = int(str[8:12])
        endY = int(str[12:16])
        return startX, startY, endX, endY

    @staticmethod
    def coord_2_str(startX, startY, endX, endY):
        """
        box coordinates  -->  string for filename
        """
        str = '{:04d}{:04d}{:04d}{:04d}'.format(startX, startY, endX, endY)
        return str

    # conversion:    sides-order (top,right,bottom,left)   <--->   corners-order  (startX,startY,endX,endY)

    @staticmethod
    def corners_2_sides(startX=None, startY=None, endX=None, endY=None
                        , corners_tuple=None):
        if corners_tuple is not None:
            (startX, startY, endX, endY) = corners_tuple
        return startY, endX, endY, startX

    @staticmethod
    def sides_2_corners(top=None, right=None, bottom=None, left=None
                        , sides_tuple=None):
        if sides_tuple is not None:
            top, right, bottom, left = sides_tuple
        return left, top, right, bottom

    # string conversion: sides-order  <---> corners-order
    # each value - 4 digits

    @staticmethod
    def corners_2_sides_str(str):
        (startX, startY, endX, endY) = Box.str_2_coord(str)
        (left, top, right, bottom) = Box.corners_2_sides(startX, startY, endX, endY)
        return Box.coord_2_str(left, top, right, bottom)

    @staticmethod
    def sides_2_corners_str(str):
        (left, top, right, bottom) = Box.str_2_coord(str)
        (startX, startY, endX, endY) = Box.sides_2_corners(left, top, right, bottom)
        return Box.coord_2_str(startX, startY, endX, endY)


class BoxesArray:
    """
    persistent (re/stored on disk) array of boxes.
    Each box is 4-item  sides-orders tuple.
    """
    def __init__(self, file_name=None):
        if file_name is None:
            self.box_arr = None
            self.counter = 0
        else:
            infile = open(file_name,'rb')
            self.box_arr = pickle.load(infile)
            infile.close()
            self.counter = self.box_arr.shape[0]
            print(f'{self.box_arr.shape[0]} corner-ordered boxes are loaded from file {file_name}')

    def add_box(self,box, allow_duplicates=False):
        if self.box_arr is None:
            self.box_arr = np.reshape(np.array(box.corners()), (4,))
        else:
            if not allow_duplicates and self.search(box,0) != -1:
                return
            self.box_arr = np.vstack( (self.box_arr,np.array(box.corners())) )
        self.counter += 1

    def save_to_disk(self,file_name):
        f = open(file_name,'wb')
        pickle.dump(self.box_arr,f)
        f.close()
        print( f'{self.box_arr.shape[0]} corner-ordered boxes are saved to file {file_name}')

    def search(self,box,eps):
        if self.counter <=2: return -1 # not fair, sorry
        deltas = self.box_arr - box.corners()
        distances = np.linalg.norm(deltas,np.inf,axis=1)
        min_ind =  np.argmin( distances )
        return min_ind if distances[min_ind] <= eps else -1

    def draw(self,frame,color=BGR_GREEN):
        """
        draw all boxes from box_arr at the frame
        """
        for b in self.box_arr:
            box=Box(corners_tuple=b)
            box.draw(frame,color)
        return frame

    def get_box(self,idx):
        """
        return Box object which is idx-th item in self.box_arr
        """
        return Box(corners_tuple= tuple(self.box_arr[idx]))


# --------------------------------------------------------------------------------------------------------

class ObjBox:

    def __init__(self, startX=None, startY=None, endX=None, endY=None, confidence=None, idx=None, label=None
                 , sides_tuple=None):
        if sides_tuple is not None:
            self.box = Box(sides_tuple=sides_tuple)
        else:
            self.box = Box(startX, startY, endX, endY)

        self.confidence = confidence if confidence is not None else -1.0
        self.idx = idx if idx is not None else -1
        self.label = label if label is not None else "unknown"

        self.face_box = None
        self.face_id = None

    def draw(self, frame, color=None):
        if color is None:
            color = BGR_GREEN
        self.box.draw(frame, color)
        label_txt = self.label if self.label is not None else ''
        confid_txt = f'd={self.confidence:.3f}' if self.confidence is not None else ''
        y = self.box.startY - 15 if self.box.startY - 15 > 15 else self.box.startY + 15
        cv2.putText(frame, f'{label_txt}  {confid_txt}', (self.box.startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def is_inside_box(self, box):
        top_box, right_box, bottom_box, left_box = box.sides()
        top, right, bottom, left = self.box.sides()
        if top >= top_box and right <= right_box and bottom <= bottom_box and left >= left_box:
            return True
        return False

    def is_inside_obj_boxes(self, obj_boxes_lst):
        for obj_box in obj_boxes_lst:
            if self.is_inside_box(obj_box.box):
                return True
        return False

    def w_h_is_in_face_range(self):
        w = self.box.endX - self.box.startX
        h = self.box.endY - self.box.startY

        if not (cfg['face_box_width_range'][0] <= w <= cfg['face_box_width_range'][1]):
            return False
        if not (cfg['face_box_height_range'][0] <= h <= cfg['face_box_height_range'][1]):
            return False
        if not (cfg['face_box_wh_ratio_range'][0] <= float(w) / float(h) <= cfg['face_box_wh_ratio_range'][1]):
            return False

        return True

    def w_h_is_in_pers_range(self):
        w = self.box.endX - self.box.startX
        h = self.box.endY - self.box.startY
        if not (cfg['pers_box_width_range'][0] <= w <= cfg['pers_box_width_range'][1]):
            return False
        if not (cfg['pers_box_height_range'][0] <= h <= cfg['pers_box_height_range'][1]):
            return False
        if not (cfg['pers_box_wh_ratio_range'][0] <= float(w) / float(h) <= cfg['pers_box_wh_ratio_range'][1]):
            return False

        return True


# -------------------------------------------------------------------------------------------------------------

def test1():
    print('testing Box class:')
    box1 = Box(10, 15, 200, 300)
    str = Box.coord_2_str(10, 15, 200, 300)
    box2 = Box(coordinate_str=str)
    if box1 != box2:
        print('error. box1={}  box2={}'.format(box1, box2))

    t, r, b, l = Box.str_2_coord(Box.corners_2_sides_str(box1.box_2_str()))
    sx, sy, ex, ey = Box.sides_2_corners(t, r, b, l)
    box3 = Box(sx, sy, ex, ey)
    if box3 != box1:
        print('error. box1={}  box3={}'.format(box1, box3))

def test2():
    b1 = Box(corners_tuple=(10,10,200,200))
    b2 = Box(corners_tuple=(30,30,420,420))
    b3 = Box(corners_tuple=(230,230,400,400))
    b_i = Box(box=b1)
    b_i.intersect(b3)
    b_u = Box(box=b1)
    b_u.union(b3)
    print(f'i: {b_i.repr()}')
    print(f'u: {b_u.repr()}')

    img = np.zeros((480, 640, 3), np.uint8)
    b1.draw(img,BGR_GREEN,'b1')
    b2.draw(img,BGR_GREEN,'b2',thickness=6)
    b3.draw(img,BGR_GREEN,'b3')
    #b_i.draw2(img,BGR_RED,'intersect')
    #b_u.draw2(img,BGR_CYAN,'union')
    (Box(box=b3).union(b2).union(b1)).draw2(img,BGR_YELLOW,'        uu')
    (Box(box=b3).intersect(b2).intersect(b1)).draw2(img,BGR_CHOCOLATE,'   ii')
    cv2.imshow('test',img)
    while cv2.waitKeyEx() != ord('q'):
        pass

if __name__ == '__main__':
    #test1()
    test2()
