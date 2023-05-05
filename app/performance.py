from ultralytics import YOLO
import face_recognition as fc
from .sort import *
import cv2
import math
import datetime
import mmcv

def performance(cap):
    model_det = YOLO('models/yolov8n.pt')
    tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
    object_id_list = []
    dtime = dict()
    dwell_time = dict()
    count = 0
    while True:
        ret, img = cap.read()
        if ret == False:
            break
        h, w, _ = img.shape
        new_w, new_h = mmcv.rescale_size((w, h), (256, np.Inf))
        img = mmcv.imresize(img, (new_w, new_h))
        if count % 2 == 0:
            all_detections = np.empty((0, 5))
            humans = model_det(img, conf=0.65, classes=0)[0].boxes.data[:, :-1]

            for human in humans:
                x1, y1, x2, y2, conf = human[0], human[1], human[2], human[3], math.ceil((human[4] * 100)) / 100
                last_detection = np.array([x1, y1, x2, y2, conf])
                all_detections = np.vstack((all_detections, last_detection))
            results_tracker = tracker.update(all_detections).tolist()

            for human in results_tracker:
                track_id, bbox = human[4], [int(human[0]), int(human[1]), int(human[2]), int(human[3])]
                if track_id not in object_id_list:
                    object_id_list.append(track_id)
                    dwell_time[track_id] = 0
                else:
                    dwell_time[track_id] += (1/15)
                    dtime[track_id] = datetime.datetime.now()
        count+=1 

    return dwell_time