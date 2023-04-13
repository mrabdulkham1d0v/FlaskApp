from flask import Flask, request, jsonify
import os
import base64
import face_recognition
from PIL import Image
from io import BytesIO
import pickle
import numpy as np

def save_embeddings_database(file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(embeddings_database, file)

def load_embeddings_database(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)

# Initialize embeddings_database by loading it from a file
embeddings_database_file = 'embeddings_database.pkl'
if os.path.exists(embeddings_database_file):
    embeddings_database = load_embeddings_database(embeddings_database_file)
else:
    embeddings_database = {}    

def generate_embedding(img):
    encodings = face_recognition.face_encodings(img)
    if len(encodings) > 0:
        return encodings[0]
    else:
        return None

def load_image_from_base64(base64_img):
    img_data = base64.b64decode(base64_img)
    img = Image.open(BytesIO(img_data))
    return np.array(img)


def add_person(person_id, front_img_base64, left_img_base64, right_img_base64):
    front_img = load_image_from_base64(front_img_base64)
    left_img = load_image_from_base64(left_img_base64)
    right_img = load_image_from_base64(right_img_base64)

    front_embedding = generate_embedding(front_img)
    left_embedding = generate_embedding(left_img)
    right_embedding = generate_embedding(right_img)
    #print(f"this is front embedding {type(front_embedding)} \n\n this is front embedding {type(left_embedding)} \n\n  this is front embedding {type(right_embedding)} \n\n ")

    if front_embedding is None or left_embedding is None or right_embedding is None:
        return False

    average_embedding = (front_embedding + left_embedding + right_embedding) / 3
    embeddings_database[person_id] = average_embedding
    # Save the updated embeddings_database to a file
    save_embeddings_database(embeddings_database_file)

    return True

def set_threshold(threshold):
    global recognition_threshold
    recognition_threshold = threshold

def recognize_person(base64_img):
    unknown_img = load_image_from_base64(base64_img)
    unknown_embedding = generate_embedding(unknown_img)
    if unknown_embedding is None:
        return None

    min_distance = float("inf")
    best_match = None

    for person_id, known_embedding in embeddings_database.items():
        distance = face_recognition.face_distance([known_embedding], unknown_embedding)

        if distance < min_distance:
            min_distance = distance
            best_match = person_id

    return best_match if min_distance <= recognition_threshold else None


recognition_threshold = 0.6

# Example usage:
# set_threshold(0.5)
# add_person("001", front_img_base64, left_img_base64, right_img_base64)
# recognized_person_id = recognize_person(unknown_img_base64)

app = Flask(__name__)

@app.route('/add_person', methods=['POST'])
def add_person_route():
    person_id = request.form['person_id']
    front_img_base64 = request.form['front_img_base64']
    left_img_base64 = request.form['left_img_base64']
    right_img_base64 = request.form['right_img_base64']
    
    result = add_person(person_id, front_img_base64, left_img_base64, right_img_base64)
    if result:
        return jsonify({'status': 'success', 'message': 'Person added successfully'})
    else:
        return jsonify({'status': 'failure', 'message': 'Failed to add person'})

@app.route('/set_threshold', methods=['POST'])
def set_threshold_route():
    threshold = float(request.form['threshold'])
    set_threshold(threshold)
    return jsonify({'status': 'success', 'message': f'Threshold set to {threshold}'})

@app.route('/recognize_person', methods=['POST'])
def recognize_person_route():
    base64_img = request.form['base64Img']
    recognized_person_id = recognize_person(base64_img)
    if recognized_person_id is not None:
        return jsonify({'status': 'success', 'person_id': recognized_person_id})
    else:
        return jsonify({'status': 'failure', 'message': 'Person not recognized'})

@app.route('/', methods=['GET'])
def get_index():
    return "Hello World!"


if __name__ == '__main__':
    app.run(debug=True)