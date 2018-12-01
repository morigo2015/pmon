# cam_detec_config

import cv2
import os

USER_DIR = '/home'+'/'+os.environ['USER']+'/'
proj_root = USER_DIR + 'mypy/cam_detect/'

_folders = {
    'video'           : proj_root + 'production/videos/' ,
    'models'          : proj_root + 'production/models/' ,
    'face_dataset'    : proj_root + 'production/face_dataset/' ,
    'event_images'    : proj_root + 'production/event_images/' ,
    'filtered_clips'  : proj_root + 'production/filtered_clips/' ,
}

_cam_fps = 4.0 # actual fps processed from cam (not sent by cam)  todo calculate automatically

cfg = {
     'show_output_frames'   : True,  # show output frames on screen (imshow(...)

    #'input_source'         : 0 ,
    #'input_source'          : _folders['video']+'bell_2018-09-25T20:55:11_ 53_0_6_0507016712190719_0680024712750719.avi' ,
    'input_source'  : 'rtsp://admin:F1123581321f_cam@192.168.1.64:554/Streaming/Channels/101' ,  # cam hik
    #'input_source' : 'rtsp://admin:F112358f_bell@192.168.1.165:554/Streaming/Channels/101' ,  # door bell
    #'input_source' : 'rtsp://admin_cam_hik:F1123581321f_cam@194.31.46.247:8554/Streaming/Channels/101' , # cam hik- remote
    #'input_source' : 'rtsp://admin:F112358f_bell@194.31.46.247:9554/Streaming/Channels/101' ,  # door bell - remote

    'input_async'           : True , # False('sync'): blocking read, True('async'): non-blocking read from frame queue
    'input_async_queue_size': 30 , # max queue size for async read
    'input_async_show_qsize': True , # include queue info in inp_frame (for debug)
    'input_async_full_delay': (1000/_cam_fps)*0.5 , # 100 , # in miliseconds, delay reading thread when queue is full
    'input_async_empty_delay': (1000/_cam_fps)*0.5 , # 100, # in miliseconds, delay main thread when queue is empty

    'input_recapture_max'   : 1000 ,  # max attempts to reopen cv2.VideoCapture
    'input_recapture_delay' : 1 , # delay in seconds between attempts to open VideCapture

    'input_show_counter_div': 10 , # debug: print frame_counter when %div ==0;  don't show if div <=0

    #'save_output_frames'    : True ,  # save output (all!) frames to file
    'save_output_frames'    : True ,  # save output (all!) frames to file
    'out_file_name'         : _folders['video']+'output.avi' ,
    'out_four_cc'           : cv2.VideoWriter_fourcc( *'XVID') ,
    'out_file_fps'          : 2*int(_cam_fps) ,

    'save_boxed_frames'     : False , # to save in videofile boxed frames
    'boxed_file_name'       : _folders['video']+'boxed.avi' ,
    'boxed_file_fps'        : 2*int(_cam_fps) ,

    'save_faced_frames'     : False,  # to save in videofile boxed frames
    'faced_file_name'       : _folders['video']+'faced.avi' ,
    'faced_file_fps'        : 4*int(_cam_fps) ,

    'obj_box_color'         : (0, 255, 0) , # greem
    'face_box_color'        : (0, 0, 255) , # red

    'pers_det_prototxt'     : _folders['models']+'MobileNetSSD_deploy.prototxt.txt' ,
    'pers_det_model'        : _folders['models']+'MobileNetSSD_deploy.caffemodel' ,

    'pers_iv'               : True , # use Intel OpenVINO
    'pers_iv_cpu_extension' : _folders['models'] + 'libcpu_extension_avx2.so', # 'libcpu_extension_sse4.so'
    'pers_iv_model'         : _folders['models'] + 'person-detection-retail-0013.xml', # 'person-detection-retail-0013.xml'
    'pers_iv_threshold'     : 0.5 , # Probability threshold for detections filtering
    'pers_det_confidence'   : 0.3 , # confidence threshold
    'pers_box_width_range'   : (30, 1200),  # (min,max) range for persbox width
    'pers_box_height_range'  : (30, 700),  # (min,max) range for persbox height
    'pers_box_wh_ratio_range': (0.25, 1./0.25),  # (min,max) range for persbox width/height

#    'pers_fake_spots_filter' : False ,
    'pers_fake_iou_threshold': 0.7 , # min IoU to be a fake_spot clip
    'pers_filtered_save'     : True,  # save filtered (fake_spots) frames into separate folder (for self-check and debug)
    'pers_filtered_save_folder': _folders['filtered_clips'],
    # 'pers_filtered_save_boxed': False,  # True if we need green boxes around fake spot

    'pers_fake_spots_file': _folders['models'] + 'fake_spots.pkl',
    'pers_fake_spots_eps'   : 8 , # distance from nearest fake_spot to treat as fake_spot too

    'facedataset_folder'    : _folders['face_dataset'],
    'encodings_file'        : _folders['face_dataset']+'face_encodings.pkl',

    #'face_needed'           : True , # include face processing: detection,encoding,classification
    'face_needed'           : False ,
    'face_det_prototxt'     : _folders['models']+'face_deploy.prototxt.txt' ,
    'face_det_model'        : _folders['models']+'face_res10_300x300_ssd_iter_140000.caffemodel' ,
    'face_det_confidence'   : 0.2 ,  # confidence threshold
    'face_box_width_range'  : (30,600) , # (min,max) range for facebox width
    'face_box_height_range' : (30,600)  , # (min,max) range for facebox height
    'face_box_wh_ratio_range':  (0.75,1./0.75) , # (min,max) range for facebox width/height

    'face_rec_use_threshold': True, # if distance to nearest neighbor < Threshold: label="Unknown"
    'face_rec_threshold'    : -1 , # -1: calculate threshold on encodings; else: the value is (manually set) threshold

    'face_filter_needed'    : False ,# check pers_box,obj_box for wh_range; check for face_box that outer pers_box exists
    'time_measure_needed'   : True ,# time measure in main loop of cam_detect

    'clips_folder'          : _folders['video'] ,  # +'clip_'
    'clips_fname_suffix'    : '.avi' ,
    'clips_include_boxes'   : False,  # include pers box (green) in clip
    'clips_prev_frames'             : 1*int(_cam_fps) , # 5 # how many previous frames include in the begin of clip
    'clips_noperson_frames_to_stop' : 3*int(_cam_fps) , # 20 # how many frames without person are allowed to not stop clip
    'clips_frames_to_appear'        : 2*int(_cam_fps) , # 10 # how many frames in clip to allow announce 'appear' event (once per clip)
    'clips_fps'                     : 2*int(_cam_fps) ,

    'event_send_message'        : True , # send telegram msg
    'event_image_folder'        : _folders['event_images'],
    'image_ext'           : '.jpg' ,

    'face_labels_list'          : ['Yulka', 'Yehor', 'Olka', 'Igor', 'Ded'] ,

# utils:
    'util_clip_folder'          : _folders['video'] ,
    'util_clip_input'           : 'clip*',

# clips-->labelled images:
    'util_raw_clips_folder'     : 'preparation/raw_clips/' ,       # здесь лежать клипы (по папкам)
    'util_img_folder'           : 'preparation/labelled_images/' , # куда складывать images (по папкам)
    'lbl_images_boxed_folder'   : 'preparation/labelled_images/_boxed_images/' ,  # for util_clip_img
    'lbl_images_boxed_needed'   : True ,

# labelled images --> encodings
    'encoding_boxed_folder'     : _folders['face_dataset']+'_boxed_images/_after_encoding/' ,
    'encoding_boxed_needed'     : True , # create and save boxed imaged when prepare embeddings

# util - test recginizer
    'test_images_folder'           : 'preparation/test_images_folder/' ,

#util - split to train,test
    'boxed_folder'              : 'preparation/labelled_images/_boxed_images/' ,
    'labelled_folder'           : 'preparation/labelled_images/' ,
    'train_folder'              : 'preparation/labelled_images/_train/' ,

    'log_file_name'             : 'log.txt',
    'exit_chars'                : [ord('q'), ord('Q'), 27]  # ord(ESC)=27
}

cfg['exi']

try:
    import cam_detect_cfg_cloud
    print(f'cloud config loaded from {cam_detect_cfg_cloud.__file__}')
except ModuleNotFoundError:
    print('cloud config is not used')
