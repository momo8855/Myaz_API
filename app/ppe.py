import cv2
import math
import time
import pandas as pd
import numpy as np
import face_recognition as fc
import datetime
import pickle
from ultralytics import YOLO
from pymongo import MongoClient
from pandas import DataFrame
from datetime import date
import cvzone
from .utils import load_measurements
from .sort import *

def get_ppe_violation(cap, collection_name, file, output_path):
    measurements_list, employee_id_list = load_measurements(file)
    saftey_equipments_names = ['Person', 'Glass', 'Helmet', 'Vest']
    ppe = ['Helmet', 'Vest']
    df = DataFrame(columns=['track_id', 'identity', 'missing', 'start', 'end'])
    model = YOLO("models/ppeY8n.pt")
    model_face = YOLO("models/faceY8n.pt")
    tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
    count = 0
    while True:
        ret, img = cap.read()
        if ret == False:
            break

        all_detections = np.empty((0, 5))

        results = model(img)
        results_faces = model_face(img, conf=0.75)
        lst_results = results[0].boxes.data.tolist()
        lst_results_faces = results_faces[0].boxes.data.tolist()
        humans = [lst for lst in lst_results if lst[5] == 0.0]
        faces = [lst for lst in lst_results_faces if lst[5] == 0]
        others = [lst for lst in lst_results if lst[5] != 0.0 and lst[4] > 0.8]

        all_detections = np.empty((0, 5))

        for human in humans:
            x1, y1, x2, y2, conf = human[0], human[1], human[2], human[3], math.ceil((human[4] * 100)) / 100

            last_detection = np.array([x1, y1, x2, y2, conf])
            all_detections = np.vstack((all_detections, last_detection))
        results_tracker = tracker.update(all_detections).tolist()
        print('Frame Tracking Complete\nStarting assigning other objects to humans')

        for result in results_tracker:
            result.append([])
            result.append([])

        for other in others:
            cx, cy = (other[2]+other[0])/2, (other[3]+other[1])/2
            for result in results_tracker:
                if result[0] < cx < result[2] and result[1] < cy < result[3]:
                    result[5].append(saftey_equipments_names[int(other[5])])
                    #result[5].append(dict(xyxy=other[:4], detection = saftey_equipments_names[int(other[5])]))

        print('Assigning objects complete\nAssigning faces to humans started')
        for face in faces:
            cx, cy = (face[2]+face[0])/2, (face[3]+face[1])/2
            for result in results_tracker:
                if result[0] < cx < result[2] and result[1] < cy < result[3]:
                    #result[5].append(saftey_equipments_names[int(other[5])])
                    result[6].append(dict(xyxy=face[:4], detection = 'face'))
        print('Assigning faces complete\nChecking today previous violations to humans started')
        for person in results_tracker:
        
            track_id, detections, face = person[4], person[5], person[6]       
            missing = [item for item in ppe if item not in detections]

            object_info = df[df.track_id == track_id].tail(1)
            if len(object_info) == 0:
                if face:
                    x1, y1, x2, y2 = face[0]['xyxy']
                    face_new = [int(y1), int(x2), int(y2), int(x1)]
                    measures = fc.face_encodings(img, [face_new])
                    matches = fc.compare_faces(measurements_list, measures[0], tolerance=0.5)
                    face_distances = fc.face_distance(measurements_list, measures[0])
                    match_index = np.argmin(face_distances)
                    if matches[match_index]:
                        identity = employee_id_list[match_index]
                    else:
                        print(face_distances[match_index])
                        identity = 'unauthorized'
                else:
                    identity = 'no_face'
                if len(missing) == 0:
                    start = None
                    end = None
                    new_row = pd.DataFrame({
                        'track_id':[track_id],
                        'identity':[identity],
                        'missing':[missing],
                        'start':[start],
                        'end':[end]
                    })
                    df = pd.concat([df, new_row], ignore_index=True)
                else:
                    start = datetime.datetime.now()
                    end = None
                    new_object = dict(arriveAt=start,  info=dict(track_id=track_id, detected=identity, endEvent=end, missing=missing), type='ppe')
                    collection_name.insert_one(new_object)
                    new_row = pd.DataFrame({
                        'track_id':[track_id],
                        'identity':[identity],
                        'missing':[missing],
                        'start':[start],
                        'end':[end]
                    })
                    df = pd.concat([df, new_row], ignore_index=True)
            else:
                if object_info.iloc[0]['identity'] == 'no_face' or object_info.iloc[0]['identity'] == 'unauthorized':
                #if object_info.iloc[0]['identity'] == 'no_face':
                    if face:
                        x1, y1, x2, y2 = face[0]['xyxy']
                        face_new = [int(y1), int(x2), int(y2), int(x1)]
                        measures = fc.face_encodings(img, [face_new])
                        matches = fc.compare_faces(measurements_list, measures[0], tolerance=0.5)
                        face_distances = fc.face_distance(measurements_list, measures[0])
                        match_index = np.argmin(face_distances)
                        if matches[match_index]:
                            identity = employee_id_list[match_index]
                        else:
                            print(face_distances[match_index])
                            identity = 'unauthorized'
                        collection_name.update_many({"info.track_id": track_id}, {'$set': {"info.detected": identity}})
                        df.loc[object_info.index[0], 'identity'] = identity
                    #else:
                    #    identity = 'no_face'
                    #df.loc[object_info.index[0], 'identity'] = identity
                    #collection_name.update_many({"info.track_id": track_id}, {'$set': {"info.detected": identity}})
                if missing:
                    if len(df.loc[object_info.index[0], 'missing']) == 0:
                        df.at[object_info.index[0], 'missing'] = missing
                        df.loc[object_info.index[0], [ 'start']] = [datetime.datetime.now()]
                        new_object = dict(arriveAt=start,  info=dict(track_id=track_id, detected=identity, endEvent=end, missing=missing), type='ppe')
                        collection_name.insert_one(new_object)
                    else:
                        if object_info.iloc[0]['missing'] == missing:
                            if (object_info.iloc[0]['end'] == None) or ((object_info.iloc[0]['end'] - datetime.datetime.now()).seconds > 60):
                                df.loc[object_info.index[0], 'end'] = datetime.datetime.now()
                                collection_name.update_one({"info.track_id": track_id}, {'$set': {"info.endEvent": datetime.datetime.now()}})
                            else:
                                start = datetime.datetime.now()
                                end = None
                                new_row = pd.DataFrame({
                                    'track_id':[track_id],
                                    'identity':[identity],
                                    'missing':[missing],
                                    'start':[start],
                                    'end':[end]
                                })
                                df = pd.concat([df, new_row], ignore_index=True)
                                new_object = dict(arriveAt=start,  info=dict(track_id=track_id, detected=identity, endEvent=end, missing=missing), type='ppe')
                                collection_name.insert_one(new_object)
                        else:
                            start = datetime.datetime.now()
                            end = None
                            new_row = pd.DataFrame({
                                'track_id':[track_id],
                                'identity':[identity],
                                'missing':[missing],
                                'start':[start],
                                'end':[end]
                            })
                            df = pd.concat([df, new_row], ignore_index=True)
                            new_object = dict(arriveAt=start,  info=dict(track_id=track_id, detected=identity, endEvent=end, missing=missing), type='ppe')
                            collection_name.insert_one(new_object)
            if face and identity != 'no_face' and identity != 'unauthorized':
                x1, y1, x2, y2 = face[0]['xyxy']
                w, h = x2 - x1, y2 - y1
                x1, y1, w, h= int(x1), int(y1), int(w), int(h)
                cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255,0,0))
                cvzone.putTextRect(img, f'{identity}', (max(0, x1), max(35, y1)), scale=0.7, thickness=1, offset=3, font=4)
                #if identity != 'unauthorized':
                #    cvzone.putTextRect(img, f'{identity}', (max(0, x1), max(35, y1)), scale=0.7, thickness=1, offset=3, font=4)
            x1, y1, x2, y2, id = person[0:5]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255,0,0))
            cvzone.putTextRect(img, f'{int(id)}', (max(0, x1), max(35, y1)), scale=0.6, thickness=1, offset=3, font=2)
            if missing:
                cvzone.putTextRect(img, f'Violation:{",".join(str(x) for x in missing)}', (max(0, x1), max(35, y1-20)), scale=1, thickness=1, offset=3, font=7)
        #for face in faces:
        #    x1, y1, x2, y2, id = face[0:5]
        #    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        #    w, h = x2 - x1, y2 - y1
        #    cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255,0,0))
        #    cvzone.putTextRect(img, f'{int(id)}', (max(0, x1), max(35, y1)), scale=2, thickness=1, offset=3)
        #for person in results_tracker:
        #    x1, y1, x2, y2, id = person[0:5]
        #    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        #    w, h = x2 - x1, y2 - y1
        #    cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255,0,0))
        #    cvzone.putTextRect(img, f'{int(id)}', (max(0, x1), max(35, y1)), scale=2, thickness=1, offset=3)
        #cv2.imshow("Image", img)
        cv2.imwrite(f'{output_path}/{count}.jpg', img)
        count+=1



def get_ppe_violation_youtube(stream, collection_name, file, output_path):
    measurements_list, employee_id_list = load_measurements(file)
    saftey_equipments_names = ['Person', 'Glass', 'Helmet', 'Vest']
    ppe = ['Helmet', 'Vest']
    df = DataFrame(columns=['track_id', 'identity', 'missing', 'start', 'end'])
    model = YOLO("models/ppeY8n.pt")
    model_face = YOLO("models/faceY8n.pt")
    tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
    count = 0
    while True:
        img= stream.read()
        if img is None:
            break

        all_detections = np.empty((0, 5))

        results = model(img)
        results_faces = model_face(img, conf=0.75)
        lst_results = results[0].boxes.data.tolist()
        lst_results_faces = results_faces[0].boxes.data.tolist()
        humans = [lst for lst in lst_results if lst[5] == 0.0]
        faces = [lst for lst in lst_results_faces if lst[5] == 0]
        others = [lst for lst in lst_results if lst[5] != 0.0 and lst[4] > 0.8]

        all_detections = np.empty((0, 5))

        for human in humans:
            x1, y1, x2, y2, conf = human[0], human[1], human[2], human[3], math.ceil((human[4] * 100)) / 100

            last_detection = np.array([x1, y1, x2, y2, conf])
            all_detections = np.vstack((all_detections, last_detection))
        results_tracker = tracker.update(all_detections).tolist()
        print('Frame Tracking Complete\nStarting assigning other objects to humans')

        for result in results_tracker:
            result.append([])
            result.append([])

        for other in others:
            cx, cy = (other[2]+other[0])/2, (other[3]+other[1])/2
            for result in results_tracker:
                if result[0] < cx < result[2] and result[1] < cy < result[3]:
                    result[5].append(saftey_equipments_names[int(other[5])])
                    #result[5].append(dict(xyxy=other[:4], detection = saftey_equipments_names[int(other[5])]))

        print('Assigning objects complete\nAssigning faces to humans started')
        for face in faces:
            cx, cy = (face[2]+face[0])/2, (face[3]+face[1])/2
            for result in results_tracker:
                if result[0] < cx < result[2] and result[1] < cy < result[3]:
                    #result[5].append(saftey_equipments_names[int(other[5])])
                    result[6].append(dict(xyxy=face[:4], detection = 'face'))
        print('Assigning faces complete\nChecking today previous violations to humans started')
        for person in results_tracker:
        
            track_id, detections, face = person[4], person[5], person[6]       
            missing = [item for item in ppe if item not in detections]

            object_info = df[df.track_id == track_id].tail(1)
            if len(object_info) == 0:
                if face:
                    x1, y1, x2, y2 = face[0]['xyxy']
                    face_new = [int(y1), int(x2), int(y2), int(x1)]
                    measures = fc.face_encodings(img, [face_new])
                    matches = fc.compare_faces(measurements_list, measures[0], tolerance=0.5)
                    face_distances = fc.face_distance(measurements_list, measures[0])
                    match_index = np.argmin(face_distances)
                    if matches[match_index]:
                        identity = employee_id_list[match_index]
                    else:
                        print(face_distances[match_index])
                        identity = 'unauthorized'
                else:
                    identity = 'no_face'
                if len(missing) == 0:
                    start = None
                    end = None
                    new_row = pd.DataFrame({
                        'track_id':[track_id],
                        'identity':[identity],
                        'missing':[missing],
                        'start':[start],
                        'end':[end]
                    })
                    df = pd.concat([df, new_row], ignore_index=True)
                else:
                    start = datetime.datetime.now()
                    end = None
                    new_object = dict(arriveAt=start,  info=dict(track_id=track_id, detected=identity, endEvent=end, missing=missing), type='ppe')
                    collection_name.insert_one(new_object)
                    new_row = pd.DataFrame({
                        'track_id':[track_id],
                        'identity':[identity],
                        'missing':[missing],
                        'start':[start],
                        'end':[end]
                    })
                    df = pd.concat([df, new_row], ignore_index=True)
            else:
                if object_info.iloc[0]['identity'] == 'no_face' or object_info.iloc[0]['identity'] == 'unauthorized':
                #if object_info.iloc[0]['identity'] == 'no_face':
                    if face:
                        x1, y1, x2, y2 = face[0]['xyxy']
                        face_new = [int(y1), int(x2), int(y2), int(x1)]
                        measures = fc.face_encodings(img, [face_new])
                        matches = fc.compare_faces(measurements_list, measures[0], tolerance=0.5)
                        face_distances = fc.face_distance(measurements_list, measures[0])
                        match_index = np.argmin(face_distances)
                        if matches[match_index]:
                            identity = employee_id_list[match_index]
                        else:
                            print(face_distances[match_index])
                            identity = 'unauthorized'
                        collection_name.update_many({"info.track_id": track_id}, {'$set': {"info.detected": identity}})
                        df.loc[object_info.index[0], 'identity'] = identity
                    #else:
                    #    identity = 'no_face'
                    #df.loc[object_info.index[0], 'identity'] = identity
                    #collection_name.update_many({"info.track_id": track_id}, {'$set': {"info.detected": identity}})
                if missing:
                    if len(df.loc[object_info.index[0], 'missing']) == 0:
                        df.at[object_info.index[0], 'missing'] = missing
                        df.loc[object_info.index[0], [ 'start']] = [datetime.datetime.now()]
                        new_object = dict(arriveAt=start,  info=dict(track_id=track_id, detected=identity, endEvent=end, missing=missing), type='ppe')
                        collection_name.insert_one(new_object)
                    else:
                        if object_info.iloc[0]['missing'] == missing:
                            if (object_info.iloc[0]['end'] == None) or ((object_info.iloc[0]['end'] - datetime.datetime.now()).seconds > 60):
                                df.loc[object_info.index[0], 'end'] = datetime.datetime.now()
                                collection_name.update_one({"info.track_id": track_id}, {'$set': {"info.endEvent": datetime.datetime.now()}})
                            else:
                                start = datetime.datetime.now()
                                end = None
                                new_row = pd.DataFrame({
                                    'track_id':[track_id],
                                    'identity':[identity],
                                    'missing':[missing],
                                    'start':[start],
                                    'end':[end]
                                })
                                df = pd.concat([df, new_row], ignore_index=True)
                                new_object = dict(arriveAt=start,  info=dict(track_id=track_id, detected=identity, endEvent=end, missing=missing), type='ppe')
                                collection_name.insert_one(new_object)
                        else:
                            start = datetime.datetime.now()
                            end = None
                            new_row = pd.DataFrame({
                                'track_id':[track_id],
                                'identity':[identity],
                                'missing':[missing],
                                'start':[start],
                                'end':[end]
                            })
                            df = pd.concat([df, new_row], ignore_index=True)
                            new_object = dict(arriveAt=start,  info=dict(track_id=track_id, detected=identity, endEvent=end, missing=missing), type='ppe')
                            collection_name.insert_one(new_object)
            if face and identity != 'no_face' and identity != 'unauthorized':
                x1, y1, x2, y2 = face[0]['xyxy']
                w, h = x2 - x1, y2 - y1
                x1, y1, w, h= int(x1), int(y1), int(w), int(h)
                cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255,0,0))
                cvzone.putTextRect(img, f'{identity}', (max(0, x1), max(35, y1)), scale=0.7, thickness=1, offset=3, font=4)
                #if identity != 'unauthorized':
                #    cvzone.putTextRect(img, f'{identity}', (max(0, x1), max(35, y1)), scale=0.7, thickness=1, offset=3, font=4)
            x1, y1, x2, y2, id = person[0:5]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255,0,0))
            cvzone.putTextRect(img, f'{int(id)}', (max(0, x1), max(35, y1)), scale=0.6, thickness=1, offset=3, font=2)
            if missing:
                cvzone.putTextRect(img, f'Violation:{",".join(str(x) for x in missing)}', (max(0, x1), max(35, y1-20)), scale=1, thickness=1, offset=3, font=7)
        #for face in faces:
        #    x1, y1, x2, y2, id = face[0:5]
        #    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        #    w, h = x2 - x1, y2 - y1
        #    cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255,0,0))
        #    cvzone.putTextRect(img, f'{int(id)}', (max(0, x1), max(35, y1)), scale=2, thickness=1, offset=3)
        #for person in results_tracker:
        #    x1, y1, x2, y2, id = person[0:5]
        #    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        #    w, h = x2 - x1, y2 - y1
        #    cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255,0,0))
        #    cvzone.putTextRect(img, f'{int(id)}', (max(0, x1), max(35, y1)), scale=2, thickness=1, offset=3)
        #cv2.imshow("Image", img)
        cv2.imwrite(f'{output_path}/{count}.jpg', img)
        count+=1
                    