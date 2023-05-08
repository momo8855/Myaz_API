from pydantic import BaseModel, EmailStr
from datetime import datetime
from typing import Optional

class UserCreate(BaseModel):
    email: EmailStr
    fname: str
    lname: str
    password: str
    permission: str = 'Basic'

class UserOut(BaseModel):
    email: EmailStr
    fname: str
    permission: str

class Token(BaseModel):
    access_token: str
    token_type: str

class Login(BaseModel):
    id: int
    fname: str
    lname: str
    email: str
    password: str
    permission: str

class Upload(BaseModel):
    file_name: str
    timestamp: datetime = datetime.now()

class TokenData(BaseModel):
    id: Optional[str] = None

class Rows(BaseModel):
    count: int

class FileName(BaseModel):
    name: str

class Path(BaseModel):
    path: str

class YoutubeLink(BaseModel):
    link:str