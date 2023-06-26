from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from dotenv import load_dotenv
from pymongo import MongoClient
import os
import base64
import face_recognition
from PIL import Image
from io import BytesIO
import numpy as np

# for testing speed
import time

# for caching embeddings
# import redis

# for serializing embeddings
import json

# Establish a connection to Redis
# redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

def get_mongo_table():
    load_dotenv()
    MONGO_URI = os.getenv("MONGO_URI")
    client = MongoClient(MONGO_URI)
    DB_NAME = os.getenv("DB_NAME")
    db = client[DB_NAME]
    TABLE_NAME = os.getenv("TABLE_NAME")
    return db[TABLE_NAME], client

users, client = get_mongo_table()


recognition_threshold = 0.4

def check_connection():
    try:
        client.server_info()
    except Exception as e:
        return False, str(e)
    else:
        return True, "Successfully connected to the database"

def generate_embedding(img):
    start_time = time.time()
    """
    Generate a face embedding from an image.
    
    Args:
    img: A numpy array representing an image.
    
    Returns:
    A face embedding as a numpy array, or None if no face could be detected in the image.
    """
    encodings = face_recognition.face_encodings(img)
    print("Time taken to face encodings image: ", time.time() - start_time)
    if len(encodings) > 0:
        return encodings[0]
    else:
        return None


def load_image_from_base64(base64_img):
    start_time = time.time()
    """
    Load an image from a base64 encoded string.
    
    Args:
    base64_img: A base64 encoded string representing an image.
    
    Returns:
    A numpy array representing the image.
    """
    img_data = base64.b64decode(base64_img)
    img = Image.open(BytesIO(img_data))
    print("Time taken to load_image_from_base64: ", time.time() - start_time)
    return np.array(img)


def set_threshold(threshold):
    """
    Set the threshold for face recognition.
    
    Args:
    threshold: A float representing the threshold.
    """
    global recognition_threshold
    recognition_threshold = threshold


# Populate Redis with user embeddings
# def populate_redis():
#     for user in users.find({}):  # iterate over all users in the MongoDB 'users' collection
#         if 'embedding' not in user or user['embedding'] is None:
#             continue
#         person_id = str(user['_id'])  # or other field that serves as person ID
#         redis_client.set(person_id, json.dumps(user['embedding']))


# Call populate_redis function at the start of your program
# populate_redis()


def recognize_person(base64_img):
    """
    Recognize a person in an image.
    
    Args:
    base64_img: A base64 encoded string representing an image.
    
    Returns:
    The ID of the recognized person, or None if no person could be recognized.
    """
    unknown_img = load_image_from_base64(base64_img)
    unknown_embedding = generate_embedding(unknown_img)
    if unknown_embedding is None:
        return None

    min_distance = float("inf")
    best_match = None
    for user in users.find({}):  # iterate over all users in the MongoDB 'users' collection
        if 'embedding' not in user or user['embedding'] is None:
            continue
        known_embedding = np.array(user['embedding'])
        person_id = str(user['_id'])  # or other field that serves as person ID

        distance = face_recognition.face_distance([known_embedding], unknown_embedding)

        if distance < min_distance:
            min_distance = distance
            best_match = person_id
    print(best_match, min_distance)
    return best_match if min_distance <= recognition_threshold else None


def write_images_to_files(images_base64):
    """
    Write base64 encoded images to separate image files.

    Args:
    images_base64: A list of base64 encoded image strings.
    """
    for i, img_base64 in enumerate(images_base64, start=1):
        img_data = base64.b64decode(img_base64)
        with open(f'image{i}.jpg', 'wb') as file:
            file.write(img_data)


app = Flask(__name__)

CORS(app, resources={r"/*": {"origins": "*", "headers": "*"}})


@app.route('/set_threshold', methods=['POST'])
@cross_origin()
def set_threshold_route():
    """
    Route for setting the threshold for face recognition.
    
    Expects a form field 'threshold' in the POST request.

    Returns:
    A JSON object with status and message fields.
    """
    threshold = float(request.form['threshold'])
    set_threshold(threshold)
    return jsonify({'status': 'success', 'message': f'Threshold set to {threshold}'})


@app.route('/get_embedding', methods=['POST'])
@cross_origin()
def get_embedding_route():
    """
    Route for getting the average face embedding from three images and adding new user to the system.
    
    Expects a JSON request with a 'user' object containing 'images' list.

    Returns:
    A JSON object with status and message fields, and the embedding if successful.
    """
    # print(request.data.keys())
    print(request.json.keys())
    user = request.json['user']
    print(user.keys())
    images_base64 = user['images']

    if len(images_base64) < 3:
        return jsonify({'status': 'failure', 'message': 'Less than 3 images received'})

    embeddings = []

    # write_images_to_files(images_base64)

    for img_base64 in images_base64:
        print("Here comes the sun !")
        img = load_image_from_base64(img_base64)
        embedding = generate_embedding(img)
        if embedding is None:
            return jsonify({'status': 'failure', 'message': 'No face detected in one of the images'})
        embeddings.append(embedding)

    average_embedding = sum(embeddings) / len(embeddings)

    # Create a new user document
    new_user = {
        "firstName": user['firstName'],
        "lastName": user['lastName'],
        "email": user['email'],
        "password": user['password'],
        "userType": user['userType'],
        "uid": user['uid'],
        "images": images_base64,
        "embedding": average_embedding.tolist()  # Store embedding
    }

    print(new_user['firstName'])

    # Insert the user document into the MongoDB 'users' collection
    result = users.insert_one(new_user)

    # Update _id of the user
    new_user['_id'] = str(result.inserted_id)

    return jsonify({'status': 'success', 'message': 'User added successfully', 'user': new_user})


@app.route('/recognize_person', methods=['POST'])
@cross_origin()
def recognize_person_route():
    """
    Route for recognizing a person in an image.
    
    Expects a form field 'base64Img' in the POST request.

    Returns:
    A JSON object with status and message fields, and the person_id if a person was recognized.
    """
    base64_img = request.form['base64Img']
    recognized_person_id = recognize_person(base64_img)
    if recognized_person_id is not None:
        return jsonify({'status': 'success', 'person_id': recognized_person_id})
    else:
        return jsonify({'status': 'failure', 'message': 'Person not recognized'})


@app.route('/', methods=['GET'])
@cross_origin()
def get_index():
    """
    The default route.

    Returns:
    A string saying 'Hello World!'.
    """
    is_connected, msg = check_connection()
    if is_connected:
        return "Hello World!! Successfully connected to the database!"
    else:
        return f"Failed to connect to the database. Error: {msg}"


if __name__ == '__main__':
    app.run(debug=True)
