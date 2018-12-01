""" Collect frames from cameras and store them to frame_queue, save to disk, display at screen
"""

import datetime
from typing import List,Tuple

import cv2
import numpy as np

# from frame_storage import Frame, frame_storage
from camera import Camera

class Frame:
    def __init__(self,cam_id:int,image:np.ndarray,time_stamp:datetime.datetime):
        self.cam_id = cam_id
        self.image = image
        self.time_stamp = time_stamp
        pass

class ToScreen:
    cancelled = False

    @classmethod
    def init(cls):
        cls.cancelled = False
        pass

    @classmethod
    def next_frame(cls,frame:Frame):
        cv2.imshow(str(frame.cam_id), frame.image)

    @classmethod
    def end_of_cycle(cls):
        ch = cv2.waitKey(1) & 0xFF
        cls.cancelled = True if ch == ord('q') or ch == 27 else False
        if cls.cancelled:
            cls.close()

    @classmethod
    def close(cls):
        cv2.destroyAllWindows()

class ToFile:
    def __init__(self):
        pass
    def next_frame(self,frame):
        pass
    def close(self):
        pass

class CamStat:
    """ Show current status of each camera"""
    msg_screen_place : List[Tuple[int,int]] = []  #   # (x,y) on screen where stat message starts
    x0 : int = None
    y0 : int = None
    y_msg_area : int = None

    @classmethod
    def init(cls, cam_list, left_top_corner_xy=(20, 1)):
        cls.x0,cls.y0 = left_top_corner_xy[0],left_top_corner_xy[1]
        cls.y_msg_area = cls.y0 + 2 + len(cam_list)
        print(f"{CamStat.esc_seq_clear()}")
        print(f"{CamStat.esc_seq_xy((cls.x0,cls.y0))}{'Id':4s} {'Name':12s} Status")
        print(f"{CamStat.esc_seq_xy((cls.x0,cls.y0+1))}{'-'*4} {'-'*12} {'-'*6}")
        for cam in cam_list:
            y=cls.y0+2+cam.cam_id
            print(f"{CamStat.esc_seq_xy((cls.x0,y))}{cam.cam_id:4d} {cam.cam_name:12s}")
            cls.msg_screen_place.append((cls.x0 + (4+1+12+1),y))
        for cam in cam_list:
            CamStat.show_msg(cam,'_'*8)

    @classmethod
    def show_msg(cls,cam, msg:str):
        print(f"{CamStat.esc_seq_xy(cls.msg_screen_place[cam.cam_id])}{msg}",end='')
        print(f"{CamStat.esc_seq_xy((cls.x0,cls.y_msg_area))}")

    @staticmethod
    def esc_seq_xy(xy: Tuple[int,int] ):
        return f"\033[s\033[{xy[1]};{xy[0]}f"

    @staticmethod
    def esc_seq_clear():
        return f"\033[2J"

    @staticmethod
    def esc_seq_save_cursor():
        return f"\033[s"

    @staticmethod
    def esc_seq_restore_cursor():
        return f"\033[u"


class VideoServer:
    to_file = None
    @classmethod
    def init(cls):
        Camera.init_cameras()
        CamStat.init(Camera.cam_list())
        for cam in Camera.cam_list():
            cam.open()
            CamStat.show_msg(cam,"connected")
            # print(f"{cam.cam_name} -- connected.")
        ToScreen.init()
        cls.to_file = ToFile()

    @classmethod
    def run(cls):
        tick_cnt = 0
        while True:
            for cam in Camera.cam_list():
                cam.get_frame()
                if not cam.read_ok:
                    #print(f"cam {cam.cam_id}: read error")
                    CamStat.show_msg(cam,f"{'Read error!!':<12s}")
                    continue
                CamStat.show_msg(cam,f"{'OK':<12s}")
                CamStat.show_msg(cam,f"{tick_cnt:>5d} frames")
                frame = Frame(cam.cam_id,cam.image,datetime.datetime.now())
                ToScreen.next_frame(frame)
                cls.to_file.next_frame(frame)
            tick_cnt += 1
            if tick_cnt%10 ==0: print(f'{tick_cnt}')
            ToScreen.end_of_cycle()
            if ToScreen.cancelled:
                cls.stop()
                return

    @classmethod
    def stop(cls):
        pass


class Display:
    """ Displays frames from cameras"""
    def init(self):
        pass

def main():
    print('start')
    VideoServer.init()
    print('***** run ******')
    print('***** run ******')
    print('***** run ******')
    print('***** run ******')
    VideoServer.run()

if __name__ == "__main__":
    main()

