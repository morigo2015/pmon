# person detection based on IntelVINO

import sys
import cv2
import numpy as np
import os
import pdb

from cam_boxes import ObjBox, Box, BoxesArray, BGR_RED, BGR_GREEN
from cam_detect_cfg import cfg
from cam_time_measure import TimeMeasure
from openvino.inference_engine import IENetwork, IEPlugin

class PersDetector:

    def __init__(self, model_xml=cfg['pers_iv_model'],
                 cpu_extension=cfg['pers_iv_cpu_extension'],
                 prob_threshold=cfg['pers_iv_threshold'],
                 device='CPU', shape=(720,1280)):
        # self.net = cv2.dnn.readNetFromCaffe(cfg['pers_det_prototxt'], cfg['pers_det_model'])
        # print(f"Loaded person detection model files:\n     proto: {cfg['pers_det_prototxt']},\n     model: {cfg['pers_det_model']}.")

        try:
            plugin_dir = None
            plugin = IEPlugin(device=device, plugin_dirs=plugin_dir)
        except RuntimeError:
            print("Error while initializing Intel OpenVINO framework")
            pdb.set_trace()
        if cpu_extension and 'CPU' in device:
            plugin.add_cpu_extension(cpu_extension)

        # Read IR
        model_bin = os.path.splitext(model_xml)[0] + ".bin"
        net = IENetwork.from_ir(model=model_xml, weights=model_bin)

        if "CPU" in plugin.device:
            supported_layers = plugin.get_supported_layers(net)
            not_supported_layers = [l for l in net.layers.keys() if l not in supported_layers]
            if len(not_supported_layers) != 0:
                print(f"Error: Following layers are not supported by the plugin for specified device {plugin.device}:\n {', '.join(not_supported_layers)}")
                print("Please try to specify cpu extensions library path in sample's command line parameters using -l or --cpu_extension command line argument")
                sys.exit(1)
        # assert len(net.inputs.keys()) == 1, "Sample supports only single input topologies"
        # assert len(net.outputs) == 1, "Sample supports only single output topologies"
        self.input_blob = next(iter(net.inputs))
        self.out_blob = next(iter(net.outputs))
        self.exec_net = plugin.load(network=net, num_requests=2)
        # Read and pre-process input image
        self.n, self.c, self.h, self.w = net.inputs[self.input_blob]
        del net
        # if args.input == 'cam':
        #     input_stream = 0
        # else:
        #     input_stream = args.input
        #     assert os.path.isfile(args.input), "Specified input file doesn't exist"
        # if args.labels:
        #     with open(args.labels, 'r') as f:
        #         labels_map = [x.strip() for x in f]
        # else:
        #     labels_map = None

        # cap = cv2.VideoCapture(input_stream)
        # out=None
        # out = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc( *'XVID'),10,(1280,720))
        # if out is None:
        #     print('error while init out')
        self.cur_request_id = 0
        # self.next_request_id = 1

        # log.info("Starting inference in async mode...")
        # log.info("To switch between sync and async modes press Tab button")
        # log.info("To stop the sample execution press Esc button")
        # is_async_mode = False # True
        # render_time = 0
        self.prob_threshold=prob_threshold
        self.initial_h, self.initial_w = shape

    def detect(self, frame):
        # ret, frame = cap.read()
        # if not ret:
        #     break
        # initial_w = cap.get(3)
        # initial_h = cap.get(4)
        in_frame = cv2.resize(frame, (self.w, self.h))
        in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
        in_frame = in_frame.reshape((self.n, self.c, self.h, self.w))

        self.exec_net.start_async(request_id=self.cur_request_id, inputs={self.input_blob: in_frame})
        r = self.exec_net.requests[self.cur_request_id].wait(-1)
        if r != 0:
            print("Error returned from inferenece wait")
            exit(1)

        res = self.exec_net.requests[self.cur_request_id].outputs[self.out_blob]
        obj_boxes = []
        for obj in res[0][0]:
            # Draw only objects when probability more than specified threshold
            confidence = obj[2]
            if confidence < self.prob_threshold:
                continue
            startX = int(obj[3] * self.initial_w)
            startY = int(obj[4] * self.initial_h)
            endX   = int(obj[5] * self.initial_w)
            endY   = int(obj[6] * self.initial_h)
            class_id = int(obj[1])
            obj_box = ObjBox(startX, startY, endX, endY, confidence, class_id, label=f'Id:{class_id}')
            obj_boxes.append(obj_box)
        return obj_boxes
    #         # Draw box and label\class_id
    #         color = (min(class_id * 12.5, 255), min(class_id * 7, 255), min(class_id * 5, 255))
    #         cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
    #         det_label = str(class_id)
    #         cv2.putText(frame, det_label + ' ' + str(round(obj[2] * 100, 1)) + ' %', (xmin, ymin - 7),
    #                     cv2.FONT_HERSHEY_COMPLEX, 0.6, color, 1)
    #
    #
    # # # Draw performance stats
    # # inf_time_message = "Inference time: N\A for async mode" if is_async_mode else \
    # #     "Inference time: {:.3f} ms".format(det_time * 1000)
    # # render_time_message = "OpenCV rendering time: {:.3f} ms".format(render_time * 1000)
    # # async_mode_message = "Async mode is on. Processing request {}".format(cur_request_id) if is_async_mode else \
    # #     "Async mode is off. Processing request {}".format(cur_request_id)
    #
    # # cv2.putText(frame, inf_time_message, (15, 15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (200, 10, 10), 1)
    # # cv2.putText(frame, render_time_message, (15, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (10, 10, 200), 1)
    # # cv2.putText(frame, async_mode_message, (10, int(initial_h - 20)), cv2.FONT_HERSHEY_COMPLEX, 0.5,
    # #             (10, 10, 200), 1)
    # # print(inf_time_message, render_time_message, async_mode_message)

    #
    # render_start = time.time()
    # cv2.imshow("Detection Results", frame)
    # out.write(frame)
    #
    # render_end = time.time()
    # render_time = render_end - render_start

    # key = cv2.waitKey(1)
    # if key == 27:
    #     break
    # if (9 == key):
    #     is_async_mode = not is_async_mode
    #     log.info("Switched to {} mode".format("async" if is_async_mode else "sync"))
    #
    # if is_async_mode:
    #     cur_request_id, next_request_id = next_request_id, cur_request_id
    #
    # out.release()
    # cv2.destroyAllWindows()
    # del exec_net
    # del plugin

    # TimeMeasure.set('  persdet-blobFromImage')
        # (h, w) = frame.shape[:2]
        # blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
        #
        # TimeMeasure.set('  persdet-forward')
        # # pass the blob through the network and obtain the detections and predictions
        # self.net.setInput(blob)
        # detections = self.net.forward()
        #
        # TimeMeasure.set('  persdet-processing')
        # obj_boxes = []
        # # loop over the detections
        # for i in np.arange(0, detections.shape[2]):
        #     # extract the confidence (i.e., probability) associated with the prediction
        #     confidence = detections[0, 0, i, 2]
        #
        #     # filter out weak detections by ensuring the `confidence` is greater than the minimum confidence
        #     if confidence < cfg['pers_det_confidence']:
        #         continue
        #
        #     # extract the index of the class label from the `detections`,
        #     # then compute the (x, y)-coordinates of the bounding box for the object
        #     idx = int(detections[0, 0, i, 1])
        #     if label != 'person':
        #         continue
        #
        #     box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        #     (startX, startY, endX, endY) = box.astype("int")
        #
        #     # if cfg['pers_fake_spots_filter'] and self.box_is_fake_spot(box,frame):
        #     #     continue # skip box if it's a fake spot
        #
        #     obj_box = ObjBox(startX, startY, endX, endY, confidence, idx, label)
        #     obj_boxes.append(obj_box)
        #
        # return obj_boxes

# -------------------------------------------------------------------------

if __name__ == "__main__":

    import glob
    import os
    from cam_boxes import Box, BGR_RED, BGR_GREEN
    from cam_time_measure import TimeMeasure

    # inp_dir = '/home/im/mypy/cam_detect/preparation/fake_spots/doorbell/'
    inp_dir = '/home/im/mypy/cam_detect/tst/'
    inp_dir = '/home/im/mypy/cam_detect/cloud/checked/event_images/persons/'
    inp_dir = '/home/im/Pictures/'
    box_dir = inp_dir+'boxed/'
    os.makedirs(box_dir,exist_ok=True)
    for s in [ f'cfg({v})={cfg[v]}' for v in cfg.keys() if 'pers_iv' in v]: print(s)

    pers_detector = PersDetector()

    for fname in glob.glob(f'{inp_dir}*.png'):
        inp_frame = cv2.imread(fname)

        TimeMeasure.set('detect')
        pers_boxes = pers_detector.detect(inp_frame)  # list of boxes with persons

        TimeMeasure.set('processing')
        for pers_box in pers_boxes:
            pers_box.draw(inp_frame, color=BGR_RED)
        cv2.imwrite(f'{box_dir}{fname.split(sep="/")[-1]}',inp_frame)
        print(f'fname={fname},  boxes={[ b.box.sides() for b in pers_boxes  ]}')

    print(TimeMeasure.results())
