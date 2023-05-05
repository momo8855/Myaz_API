from fastapi import FastAPI, Response, status, HTTPException, Depends, APIRouter
from .. import schemas, oauth2, utils
from ..database import get_db, get_mongo_db
from datetime import datetime
import psycopg2
from ..attendance import mark_attendance
from ..ppe import get_ppe_violation
import cv2
import os
import ffmpeg
from ..action import get_actions
from ..performance import performance

router = APIRouter(
    prefix="/task",
    tags=['task']
)

@router.post("/attendance", status_code=status.HTTP_200_OK)
async def take_attendance(path: schemas.Path, db = Depends(get_db), current_user: int = Depends(oauth2.get_current_user)):
    dbname = get_mongo_db('main-data')
    collection_name = dbname["employees"]
    collection_name_attendance = dbname["attendances_till_fixs"]
    file = 'app\EncodeFile.p'
    cap = cv2.VideoCapture(path.path)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print( length )
    ids = mark_attendance(cap=cap, collection_name=collection_name, collection_name_attendance=collection_name_attendance, file=file)
    cursor, conn = db
    name = 'attendance'
    type = 'ai'
    cursor.execute("""INSERT INTO "Task" (name, type, date, user_id) VALUES(%s, %s, %s, %s)""", (
            name, type, datetime.now(), current_user.id
        ))
    conn.commit()
    cursor.close()
    conn.close()

    return f"employees with {ids} were successfully marked for today"

@router.post("/ppe", status_code=status.HTTP_200_OK)
async def take_attendance(path: schemas.Path, db = Depends(get_db), current_user: int = Depends(oauth2.get_current_user)):
    dbname = get_mongo_db('main-data')
    collection_name = dbname["events"]
    file = 'app\EncodeFile.p'
    cap = cv2.VideoCapture(path.path)
    os.makedirs(f"{current_user.id}_frames", exist_ok=True)
    output_path = f"{current_user.id}_frames"
    get_ppe_violation(cap=cap, collection_name=collection_name, file=file, output_path=output_path)

    input_pattern = f'{output_path}/%d.jpg'
    os.makedirs(f"{current_user.id}_processed", exist_ok=True)
    output_file = f"{current_user.id}_processed/{os.path.basename(path.path)}"

    ffmpeg.input(input_pattern, framerate=30).output(output_file, codec='libx264', pix_fmt='yuv420p', crf=23, preset='slow').run()

    cursor, conn = db
    name = 'ppe'
    type = 'ai'
    cursor.execute("""INSERT INTO "Task" (name, type, date, user_id) VALUES(%s, %s, %s, %s)""", (
            name, type, datetime.now(), current_user.id
        ))
    conn.commit()
    cursor.close()
    conn.close()

    return f"{output_file} was successfully create"

@router.post("/action", status_code=status.HTTP_200_OK)
async def get_evenvts(path: schemas.Path, db = Depends(get_db), current_user: int = Depends(oauth2.get_current_user)):
    dbname = get_mongo_db('main-data')
    collection_name = dbname["events"]
    file = 'app\EncodeFile.p'
    cap = cv2.VideoCapture(path.path)
    get_actions(cap=cap, collection_name=collection_name, file=file)

@router.post("/performance", status_code=status.HTTP_200_OK)
async def get_performance(path: schemas.Path, db = Depends(get_db), current_user: int = Depends(oauth2.get_current_user)):
    cap = cv2.VideoCapture(path.path)
    seconds = performance(cap=cap)

    return seconds