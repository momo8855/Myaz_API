from fastapi import File, UploadFile, Response, status, HTTPException, Depends, APIRouter
from fastapi.responses import FileResponse
from .. import schemas, oauth2, utils
from ..database import get_db
import os
import datetime
import psycopg2
import shutil
import mimetypes

router = APIRouter(
    prefix="/file",
    tags=['File']
)

@router.post("/uploadvideo", status_code=status.HTTP_201_CREATED, response_model=schemas.Upload)
async def create_upload_video(video: UploadFile = File(...), db = Depends(get_db), current_user: int = Depends(oauth2.get_current_user)):
    # Check if video name already exists
    cursor, conn = db
    cursor.execute("""SELECT name FROM "File" WHERE id = %s and name = %s""", (current_user.id,video.filename))
    name = cursor.fetchone()
    if name:
        name = schemas.FileName(**name)
        if name == video.filename:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT, detail=f"File already exists please rename your file")
    # Get number of videos for this user
    cursor.execute("""SELECT COUNT(*) FROM "File" WHERE id = %s""", (current_user.id,))
    files = cursor.fetchone()
    files = schemas.Rows(**files)
    if files.count > 5:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail=f"You have exceeded your 5 files limit. Please delete a file and try again")
    # Get video size
    video_size = await utils.get_video_size(video)
    print(video_size)
    if video_size > 50:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail=f"You can only upload files less than 50 mb")
    # Insert into the database that this user uploaded a video
    name, extension, path, user_id = os.path.splitext(video.filename)[0], video.filename.split(".")[-1], f"{current_user.id}/{video.filename}", current_user.id
    try:
        cursor.execute("""INSERT INTO "File" (name, extension, type, path, user_id, date) VALUES(%s, %s, %s, %s, %s, %s)""", (
                name, extension, 'video', path, user_id, datetime.datetime.now()
            ))
        conn.commit()
    except psycopg2.errors.StringDataRightTruncation as e:
        raise HTTPException(status_code=400, detail='Please rename your file to be max 50 character')
    
    # Create the videos folder for this user if it doesn't exist
    await video.seek(0)
    os.makedirs(f"{current_user.id}_uploaded", exist_ok=True)
    # Save the video to the "videos" folder
    file_path = f"{current_user.id}_uploaded/{video.filename}"
    with open(file_path, "wb+") as file_object:
      file_object.write(video.file.read())

    cursor.close()
    conn.close()
    
    
    return dict(file_name = video.filename)


@router.get("/download", status_code=status.HTTP_200_OK)
async def download_video(path: schemas.Path, db = Depends(get_db), current_user: int = Depends(oauth2.get_current_user)):
    if not os.path.exists(path.path):
        raise HTTPException(status_code=404, detail="Video file not found") # raise a 404 HTTP exception if the file does not exist
    mime_type, _ = mimetypes.guess_type(path.path)
    if mime_type is None:
        mime_type = "application/octet-stream"  # use a default media type if the mime type is unknown

    return FileResponse(path.path, media_type=mime_type, filename="video.mp4")


@router.post("/delete", status_code=status.HTTP_201_CREATED)
async def delete_all(db = Depends(get_db), current_user: int = Depends(oauth2.get_current_user)):
    folder_names = [f'{current_user.id}_frames', f'{current_user.id}_uploaded', f'{current_user.id}_processed']  # list the names of the folders you want to delete
    for folder in folder_names:
        try:
            shutil.rmtree(folder)  # remove the folder using the rmdir() function
            print(f"{folder} has been deleted.")  # print a message to confirm the folder has been deleted
        except OSError as e:
            print(f"Error: {folder} could not be deleted - {e}")  # print an error message if the folder could not be deleted

    cursor, conn = db
    cursor.execute("""Delete FROM "File" WHERE user_id = %s""", (current_user.id,))
    conn.commit()
    cursor.close()
    conn.close()

    return dict(message="Files were successfully deleted if they existed")