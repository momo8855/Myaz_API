from passlib.context import CryptContext
import pickle
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash(password: str):
    return pwd_context.hash(password)

def verify(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

async def get_video_size(video):
    size = 0
    while True:
        chunk = await video.read(1024)
        if not chunk:
            break
        size += len(chunk)
    size = size / (1024 * 1024)

    return size

def load_measurements(filename):
    # Read measures from pickle file
    measurements_with_id = []
    with open(filename, 'rb') as file:
        measurements_with_id = pickle.load(file)

    # Split the measures and ids
    measurements_list, employee_id_list = measurements_with_id
    return measurements_list, employee_id_list