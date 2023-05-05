import math
import pandas as pd
from .sort import *
import pickle
import face_recognition as fc
import random
from pymongo import MongoClient
import datetime

from .utils import load_measurements

import cvzone
import cv2
import mmcv
import mmengine
import numpy as np
import torch
from mmengine import DictAction
from mmengine.runner import load_checkpoint
from mmengine.structures import InstanceData

from mmaction.apis import detection_inference
from mmaction.registry import MODELS
from mmaction.structures import ActionDataSample
from ultralytics import YOLO

def load_label_map(file_path):
    """Load Label Map.

    Args:
        file_path (str): The file path of label map.
    Returns:
        dict: The label map (int -> label name).
    """
    lines = open(file_path).readlines()
    lines = [x.strip().split(': ') for x in lines]
    return {int(x[0]): x[1] for x in lines}

def get_actions(cap, collection_name, file):
    measurements_list, employee_id_list = load_measurements(file)
    cfg_options={}
    config = mmengine.Config.fromfile('configs\\detection\\slowonly\\slowonly_kinetics400-pretrained-r101_8xb16-8x8x1-20e_ava21-rgb.py')
    config.merge_from_dict(cfg_options)
    val_pipeline = config.val_pipeline
    label_map='label_map.txt'

    label_map='label_map.txt'
    # Load label_map
    label_map = load_label_map(label_map)
    try:
        if config['data']['train']['custom_classes'] is not None:
            label_map = {
                id + 1: label_map[cls]
                for id, cls in enumerate(config['data']['train']
                                         ['custom_classes'])
            }
    except KeyError:
        pass

    device='cpu'
    # Build STDET model
    try:
        # In our spatiotemporal detection demo, different actions should have
        # the same number of bboxes.
        config['model']['test_cfg']['rcnn'] = dict(action_thr=0)
    except KeyError:
        pass
    config.model.backbone.pretrained = None
    model = MODELS.build(config.model)
    load_checkpoint(model, 'checkpoint\\slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb_20201217-16378594.pth', map_location='cpu')
    model.to(device)
    model.eval()

    predictions = []
    img_norm_cfg = dict(
        mean=np.array(config.model.data_preprocessor.mean),
        std=np.array(config.model.data_preprocessor.std),
        to_rgb=False)
    model_det = YOLO('models/yolov8n.pt')
    model_face = YOLO('models/faceY8n.pt')
    df = pd.DataFrame({
        'track_id':[],
        'identity':[],
        'action':[],
        'start':[],
        'end':[]    
    })
    tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
    count = 0
    imgs_original=[]
    imgs=[]
    while True:
        ret, img = cap.read()
        if ret == False:
            break
        h, w, _ = img.shape
        new_w, new_h = mmcv.rescale_size((w, h), (256, np.Inf))
        img = mmcv.imresize(img, (new_w, new_h))
    
        if count % 8 == 0:
            print(count // 8)
            img_copy = img
            img_copy = img_copy.astype(np.float32)
            _ = mmcv.imnormalize_(img_copy, **img_norm_cfg)
            imgs.append(img_copy)
            imgs_original.append(img)

            all_detections = np.empty((0, 5))
            humans = model_det(img, conf=0.65, classes=0)
            faces = model_face(img, conf=0.65)[0].boxes.data.tolist()
            humans = humans[0].boxes.data[:, :-1]
            for human in humans:
                x1, y1, x2, y2, conf = human[0], human[1], human[2], human[3], math.ceil((human[4] * 100)) / 100
                last_detection = np.array([x1, y1, x2, y2, conf])
                all_detections = np.vstack((all_detections, last_detection))
            results_tracker = tracker.update(all_detections).tolist()
            tensor_tracker = torch.tensor(results_tracker)

            for result in results_tracker:
                result.append([])
                result.append([])

            for face in faces:
                cx, cy = (face[2]+face[0])/2, (face[3]+face[1])/2
                for result in results_tracker:
                    if result[0] < cx < result[2] and result[1] < cy < result[3]:
                        result[5].append(face[:4])

            if len(imgs) == 8:
                if tensor_tracker.shape[0] == 0:
                    predictions.append(None)
                    imgs = []
                    imgs_original=[]
                    continue
                tensor_tracker = tensor_tracker[:, :-1]
                input_array = np.stack(imgs).transpose((3, 0, 1, 2))[np.newaxis]
                input_tensor = torch.from_numpy(input_array).to(device)

                datasample = ActionDataSample()
                datasample.proposals = InstanceData(bboxes=tensor_tracker)
                datasample.set_metainfo(dict(img_shape=(new_h, new_w)))

                with torch.no_grad():
                    result = model(input_tensor, [datasample], mode='predict')
                    imgs = []
                    imgs_original=[]
                    scores = result[0].pred_instances.scores
                    print(scores)
                    prediction = []
                    # N proposals
                    for i in range(tensor_tracker.shape[0]):
                        prediction.append([results_tracker[i][4], [label_map[64], scores[i][64]] , [label_map[54], scores[i][54]], [label_map[15], scores[i][15]]])

                    for i in range(len(results_tracker)):
                        if prediction[i][1][1] > 0.1:
                            results_tracker[i][6].append('fight')
                        if prediction[i][2][1] > 0.3:
                            results_tracker[i][6].append('smoke')
                        if prediction[i][3][1] > 0.5:
                            results_tracker[i][6].append('phone')

                    print(results_tracker)
                    predictions.append(prediction)

                    for person in results_tracker:                 
                        track_id, face, action = person[4], person[5], person[6]
                        object_info = df[df.track_id == track_id].tail(1)
                        if len(object_info) == 0:
                            if face:
                                x1, y1, x2, y2 = face[0]
                                face = [int(y1), int(x2), int(y2), int(x1)]
                                measures = fc.face_encodings(img, [face])
                                matches = fc.compare_faces(measurements_list, measures[0])
                                face_distances = fc.face_distance(measurements_list, measures[0])
                                match_index = np.argmin(face_distances)
                                if matches[match_index]:
                                    identity = employee_id_list[match_index]

                                else:
                                    identity = 'unauthorized'
                            else:
                                identity = 'no_face'
                            if len(action) == 0:
                                start = None
                                end = None
                                new_row = pd.DataFrame({
                                    'track_id':[track_id],
                                    'identity':[identity],
                                    'action':[action],
                                    'start':[start],
                                    'end':[end]
                                })
                                df = pd.concat([df, new_row], ignore_index=True)
                            else:
                                start = datetime.datetime.now()
                                end = None
                                new_object = dict(arriveAt=start,  info=dict(track_id=track_id, detected=identity, endEvent=end, cam=random.randint(0, 9)), type=action[0])
                                collection_name.insert_one(new_object)
                                new_row = pd.DataFrame({
                                    'track_id':[track_id],
                                    'identity':[identity],
                                    'action':[action[0]],
                                    'start':[start],
                                    'end':[end]
                                })
                                df = pd.concat([df, new_row], ignore_index=True)
                        else:
                            if object_info.iloc[0]['identity'] == 'no_face':
                                if face:
                                    x1, y1, x2, y2 = face[0]
                                    face = [int(y1), int(x2), int(y2), int(x1)]
                                    measures = fc.face_encodings(img, [face])
                                    matches = fc.compare_faces(measurements_list, measures[0])
                                    face_distances = fc.face_distance(measurements_list, measures[0])
                                    match_index = np.argmin(face_distances)
                                    if matches[match_index]:
                                        identity = employee_id_list[match_index]

                                    else:
                                        identity = 'unauthorized'
                                    print(identity)
                                    print('changed')
                                    collection_name.update_many({"info.track_id": track_id}, {'$set': {"info.detected": identity}})
                                    df.loc[object_info.index[0], 'identity'] = identity
                                else:
                                    identity = 'no_face'
                                df.loc[object_info.index[0], 'identity'] = identity
                                collection_name.update_many({"info.track_id": track_id}, {'$set': {"info.detected": identity}})
                            if action:
                                if len(df.loc[object_info.index[0], 'action']) == 0:
                                    df.loc[object_info.index[0], ['action', 'start']] = [action[0], datetime.datetime.now()]
                                    print('first')
                                    print(df.loc[object_info.index[0]])
                                    new_object = dict(arriveAt=datetime.datetime.now(),  info=dict(track_id=track_id, detected=identity, endEvent=end, cam=random.randint(0, 9)), type=action[0])
                                    collection_name.insert_one(new_object)

                                else:
                                    if object_info.iloc[0]['action'] == action[0]:
                                        if (object_info.iloc[0]['end'] == None) or ((object_info.iloc[0]['end'] - datetime.datetime.now()).seconds > 60):
                                            df.loc[object_info.index[0], 'end'] = datetime.datetime.now()
                                            collection_name.update_one({"info.track_id": track_id}, {'$set': {"info.endEvent": datetime.datetime.now()}})                 
                                        else:
                                            start = datetime.datetime.now()
                                            end = None
                                            new_row = pd.DataFrame({
                                                'track_id':[track_id],
                                                'identity':[identity],
                                                'action':[action[0]],
                                                'start':[start],
                                                'end':[end]
                                            })
                                            df = pd.concat([df, new_row], ignore_index=True)
                                            new_object = dict(arriveAt=start,  info=dict(track_id=track_id, detected=identity, endEvent=end, cam=random.randint(0, 9)), type=action[0])
                                            collection_name.insert_one(new_object)                    
                                    else:
                                        start = datetime.datetime.now()
                                        end = None
                                        new_row = pd.DataFrame({
                                            'track_id':[track_id],
                                            'identity':[identity],
                                            'action':[action[0]],
                                            'start':[start],
                                            'end':[end]
                                        })
                                        df = pd.concat([df, new_row], ignore_index=True)
                                        new_object = dict(arriveAt=start,  info=dict(track_id=track_id, detected=identity, endEvent=end, cam=random.randint(0, 9)), type=action[0])
                                        collection_name.insert_one(new_object)

        #for person in results_tracker:
        #    x1, y1, x2, y2, id = person[0:5]
        #    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        #    w, h = x2 - x1, y2 - y1
        #    cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255,0,0))
        #    cvzone.putTextRect(img, f'{int(id)}', (max(0, x1), max(35, y1)), scale=2, thickness=1, offset=3)
        #cv2.imshow("Image", img)
        #if cv2.waitKey(10) & 0xFF == ord('q'):
        #    break
        count+=1  
    #cap.release()
    #cv2.destroyAllWindows() 
