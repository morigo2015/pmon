# Event manager

import datetime

from cam_detect_cfg import cfg

class EventManager:

    def __init__(self):
        self.last_appeared_time = None
        pass

    def get_event(self, obj_boxes, face_boxes):
        if len(obj_boxes)==0 and len(face_boxes)==0: return 'nothing', 'no msg'
        now = datetime.datetime.now()
        if self.last_appeared_time is None \
            or (now - self.last_appeared_time).seconds > cfg['event_nomsg_interval']: # exec only if is not None
            self.last_appeared_time = now
            message = '{}\n{}\n'.format(now,'undef person', )
            message += 'confid:'
            if len(obj_boxes) >0: message += ' pers={:.2f}'.format(obj_boxes[0].confidence) # !!!!*******!!!! later
            if len(face_boxes)>0: message += ' face={:.2f}'.format(face_boxes[0].confidence) # !!!!*******!!!! later
            return 'appeared',message
        else:
            return 'nothing','no message'

    def log_event(self, event, obj_box):
        print( '{}:  : event={} conf={:.2f}'.format(datetime.datetime.now(),event, obj_box.confidence))
        pass