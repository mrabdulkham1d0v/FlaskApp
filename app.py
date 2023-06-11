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

load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")

client = MongoClient(MONGO_URI)
db = client.test
users = db.users

recognition_threshold = 0.4

def generate_embedding(img):
    """
    Generate a face embedding from an image.
    
    Args:
    img: A numpy array representing an image.
    
    Returns:
    A face embedding as a numpy array, or None if no face could be detected in the image.
    """
    encodings = face_recognition.face_encodings(img)
    if len(encodings) > 0:
        return encodings[0]
    else:
        return None

def load_image_from_base64(base64_img):
    """
    Load an image from a base64 encoded string.
    
    Args:
    base64_img: A base64 encoded string representing an image.
    
    Returns:
    A numpy array representing the image.
    """
    img_data = base64.b64decode(base64_img)
    img = Image.open(BytesIO(img_data))
    return np.array(img)

def set_threshold(threshold):
    """
    Set the threshold for face recognition.
    
    Args:
    threshold: A float representing the threshold.
    """
    global recognition_threshold
    recognition_threshold = threshold

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

    for user in users.find({}):   # iterate over all users in the MongoDB 'users' collection
        if 'embedding' not in user or user['embedding'] is None: 
            continue
        known_embedding = np.array(user['embedding'])
        person_id = str(user['_id'])   # or other field that serves as person ID

        distance = face_recognition.face_distance([known_embedding], unknown_embedding)

        if distance < min_distance:
            min_distance = distance
            best_match = person_id

    return best_match if min_distance <= recognition_threshold else None


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
    Route for getting the average face embedding from three images.
    
    Expects form fields 'front_img_base64', 'left_img_base64', and 'right_img_base64' in the POST request.

    Returns:
    A JSON object with status and message fields, and the embedding if successful.
    """
    front_img_base64 = request.form['front_img_base64']
    left_img_base64 = request.form['left_img_base64']
    right_img_base64 = request.form['right_img_base64']
    
    front_img = load_image_from_base64(front_img_base64)
    left_img = load_image_from_base64(left_img_base64)
    right_img = load_image_from_base64(right_img_base64)

    front_embedding = generate_embedding(front_img)
    left_embedding = generate_embedding(left_img)
    right_embedding = generate_embedding(right_img)

    if front_embedding is None or left_embedding is None or right_embedding is None:
        return jsonify({'status': 'failure', 'message': 'No face detected in one of the images'})

    average_embedding = (front_embedding + left_embedding + right_embedding) / 3
    return jsonify({'status': 'success', 'embedding': average_embedding.tolist()})

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
    return "Hello World!"


if __name__ == '__main__':
    app.run(debug=True)
