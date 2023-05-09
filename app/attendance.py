import numpy as np
import face_recognition as fc
from datetime import datetime, date
import datetime
import pickle
import cv2
from pymongo import MongoClient
from random import randint, randrange
from .utils import load_measurements


def mark_attendance(cap, collection_name, collection_name_attendance, file):
    measurements_list, employee_id_list = load_measurements(file)
    ids=[]
    while cap.isOpened():
        ret, img = cap.read()
        if ret == False:
            break
        img_resize = cv2.resize(img, (352, 624))
        img_smaller = cv2.resize(img_resize, (0, 0), None, 0.4, 0.4)
        img_smaller = cv2.cvtColor(img_smaller, cv2.COLOR_BGR2RGB)
        detected_faces = fc.face_locations(img_smaller)
        measures = fc.face_encodings(img_smaller, detected_faces)
        if detected_faces:
            for face_measure, face_location in zip(measures, detected_faces):
                matches = fc.compare_faces(measurements_list, face_measure)
                face_distances = fc.face_distance(measurements_list, face_measure)
                print(face_distances)
                match_index = np.argmin(face_distances)
                if matches[match_index]:
                    id = employee_id_list[match_index]
                    if id not in ids:
                        ids.append(id)
                    employee = collection_name.find_one(dict(employee_id = id), dict(attendance ={'$slice':-1}, _id = 0, arrivedAt = 0, hireDate = 0, violations = 0, gender = 0, description = 0, employee_id = 0))
                    last_attendance = employee['attendance'][0]
                    time_elapsed = (datetime.datetime.today() - last_attendance).seconds
                    if time_elapsed > 60:
                        collection_name.update_one(dict(employee_id = id), {'$push':dict(attendance = datetime.datetime.now())})
                        today = collection_name_attendance.find_one(dict(date = datetime.datetime.today().strftime("%d-%m-%Y")))
                        if today:
                            # If today exist add attendance
                            collection_name_attendance.update_one(dict(date = datetime.datetime.today().strftime("%d-%m-%Y")), {'$push':dict(employees = dict(name= employee['name'],department= employee['department'], job = employee['job'], employee_id = id,arrive_at = datetime.datetime.now()))})
                        else:
                            # Else create day then add
                            collection_name_attendance.insert_one(dict(date = datetime.datetime.today().strftime("%d-%m-%Y"), employees=[]))
                            collection_name_attendance.update_one(dict(date = datetime.datetime.today().strftime("%d-%m-%Y")), {'$push':dict(employees = dict(name= employee['name'],department= employee['department'], job = employee['job'], employee_id = id,arrive_at = datetime.datetime.now()))})
                                     
    return ids

def mark_attendance_youtube(stream, collection_name, collection_name_attendance, file):
    measurements_list, employee_id_list = load_measurements(file)
    ids=[]
    while True:
        img = stream.read()
        if img is None:
            break
        img_resize = cv2.resize(img, (352, 624))
        img_smaller = cv2.resize(img_resize, (0, 0), None, 0.4, 0.4)
        img_smaller = cv2.cvtColor(img_smaller, cv2.COLOR_BGR2RGB)
        detected_faces = fc.face_locations(img_smaller)
        measures = fc.face_encodings(img_smaller, detected_faces)
        if detected_faces:
            for face_measure, face_location in zip(measures, detected_faces):
                matches = fc.compare_faces(measurements_list, face_measure)
                face_distances = fc.face_distance(measurements_list, face_measure)
                match_index = np.argmin(face_distances)
                if matches[match_index]:
                    id = employee_id_list[match_index]
                    if id not in ids:
                        ids.append(id)
                    employee = collection_name.find_one(dict(employee_id = id), dict(attendance ={'$slice':-1}, _id = 0, arrivedAt = 0, hireDate = 0, violations = 0, gender = 0, description = 0, employee_id = 0))
                    last_attendance = employee['attendance'][0]
                    time_elapsed = (datetime.datetime.today() - last_attendance).seconds
                    if time_elapsed > 60:
                        collection_name.update_one(dict(employee_id = id), {'$push':dict(attendance = datetime.datetime.now())})
                        today = collection_name_attendance.find_one(dict(date = datetime.datetime.today().strftime("%d-%m-%Y")))
                        if today:
                            # If today exist add attendance
                            collection_name_attendance.update_one(dict(date = datetime.datetime.today().strftime("%d-%m-%Y")), {'$push':dict(employees = dict(name= employee['name'],department= employee['department'], job = employee['job'], employee_id = id,arrive_at = datetime.datetime.now()))})
                        else:
                            # Else create day then add
                            collection_name_attendance.insert_one(dict(date = datetime.datetime.today().strftime("%d-%m-%Y"), employees=[]))
                            collection_name_attendance.update_one(dict(date = datetime.datetime.today().strftime("%d-%m-%Y")), {'$push':dict(employees = dict(name= employee['name'],department= employee['department'], job = employee['job'], employee_id = id,arrive_at = datetime.datetime.now()))})
                                     
    return ids