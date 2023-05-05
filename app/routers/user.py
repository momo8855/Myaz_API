from fastapi import FastAPI, Response, status, HTTPException, Depends, APIRouter
from fastapi.security.oauth2 import OAuth2PasswordRequestForm
from .. import schemas, utils, oauth2
from ..database import get_db
from datetime import datetime
import psycopg2


router = APIRouter(
    prefix="/user",
    tags=['Users']
)

@router.post("/signup", status_code=status.HTTP_201_CREATED, response_model=schemas.UserOut)
def create_user(user: schemas.UserCreate, db=Depends(get_db)):
    hashed_password = utils.hash(user.password)
    user.password = hashed_password
    cursor, conn = db
    user.permission = 'Basic'
    try:
        cursor.execute("""INSERT INTO "User" (fname, lname, email, password, permission, created_at) VALUES(%s, %s, %s, %s, %s, %s)""", (
            user.fname, user.lname, user.email, user.password, user.permission, datetime.now()
        ))
        conn.commit()
    except psycopg2.errors.UniqueViolation:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT,
                            detail="User already exists")
    
    cursor.close()
    conn.close()
    return user

@router.post("/login", response_model=schemas.Token)
def login(user_credentials: OAuth2PasswordRequestForm = Depends(), db = Depends(get_db)):
    cursor, conn = db
    cursor.execute("""SELECT * FROM "User" WHERE email = %s""", (user_credentials.username,))
    user = cursor.fetchone()
    user = schemas.Login(**user)
    

    if not user:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail=f"Invalid Credentials")

    if not utils.verify(user_credentials.password, user.password):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail=f"Invalid Credentials")
    
    access_token = oauth2.create_access_token(data={"user_id": user.id, "permission": user.permission})
    cursor.close()
    conn.close()

    return {"access_token": access_token, "token_type": "bearer"}