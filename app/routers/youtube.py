from fastapi import FastAPI, Response, status, HTTPException, Depends, APIRouter
from .. import schemas, oauth2, utils
from ..database import get_db, get_mongo_db
from datetime import datetime
import psycopg2
from ..attendance import mark_attendance_youtube
from ..ppe import get_ppe_violation_youtube
import cv2
import os
import ffmpeg
from ..action import get_actions_youtube
from ..performance import performance_youtube
from vidgear.gears import CamGear

router = APIRouter(
    prefix="/youtube",
    tags=['YouTube']
)

@router.post("/action", status_code=status.HTTP_200_OK)
async def get_evenvts_camera(link: schemas.YoutubeLink, db = Depends(get_db), current_user: int = Depends(oauth2.get_current_user)):
    dbname = get_mongo_db('main-data')
    collection_name = dbname["events"]
    file = 'app\EncodeFile.p'
    options = {"STREAM_RESOLUTION": "480p", "CAP_PROP_FPS":20}
    stream = CamGear(
        source= link.link,
        stream_mode=True,
        logging=True,
        **options
    ).start()
    get_actions_youtube(stream=stream, collection_name=collection_name, file=file)


@router.post("/performance", status_code=status.HTTP_200_OK)
async def get_performance(link: schemas.YoutubeLink, db = Depends(get_db), current_user: int = Depends(oauth2.get_current_user)):
    options = {"STREAM_RESOLUTION": "480p", "CAP_PROP_FPS":20}
    stream = CamGear(
        source= link.link,
        stream_mode=True,
        logging=True,
        **options
    ).start()
    seconds = performance_youtube(stream=stream)

    return seconds

@router.post("/attendance", status_code=status.HTTP_200_OK)
async def take_attendance(link: schemas.YoutubeLink, db = Depends(get_db), current_user: int = Depends(oauth2.get_current_user)):
    dbname = get_mongo_db('main-data')
    collection_name = dbname["employees"]
    collection_name_attendance = dbname["attendances_till_fixs"]
    file = 'app\EncodeFile.p'
    options = {"STREAM_RESOLUTION": "480p", "CAP_PROP_FPS":20}
    stream = CamGear(
        source= link.link,
        stream_mode=True,
        logging=True,
        **options
    ).start()
    ids = mark_attendance_youtube(stream=stream, collection_name=collection_name, collection_name_attendance=collection_name_attendance, file=file)
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
async def get_violation(path: schemas.Path, db = Depends(get_db), current_user: int = Depends(oauth2.get_current_user)):
    dbname = get_mongo_db('main-data')
    collection_name = dbname["events"]
    file = 'app\EncodeFile.p'
    cap = cv2.VideoCapture(path.path)
    os.makedirs(f"{current_user.id}_frames", exist_ok=True)
    output_path = f"{current_user.id}_frames"
    get_ppe_violation_youtube(cap=cap, collection_name=collection_name, file=file, output_path=output_path)

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