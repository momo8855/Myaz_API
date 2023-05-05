import psycopg2 as psy
from psycopg2.extras import RealDictCursor
from .config import settings
import time
from pymongo import MongoClient


def get_db():
    while True:
        try:
            conn = psy.connect(host=settings.database_hostname, database=settings.database_name, user=settings.database_username,
                               password=settings.database_password, port=settings.database_port, cursor_factory=RealDictCursor)
            cursor = conn.cursor()
            print("Database Connection was Succesful")
            break
        except Exception as error:
            print("Connection to Database Field")
            print("Error: ", error)
            time.sleep(2)
    return cursor, conn

def get_mongo_db(db_name):
    client = MongoClient('mongodb+srv://mostafaMohsen:u2vp4JpU9CfcF@cluster0.evxfmf1.mongodb.net/?retryWrites=true&w=majority')
    return client[db_name]
