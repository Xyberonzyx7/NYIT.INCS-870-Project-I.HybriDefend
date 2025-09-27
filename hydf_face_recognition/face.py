import face_recognition
import os
import json
from PIL import Image
import numpy as np

module_dir = os.path.dirname(os.path.abspath(__file__))

names = ['David', 'Kevin']

def get_name_label():
    return names


def load_encodings_from_folder(name, folder_path):
    """
    Loads all encodings from a folder containing images of a single person.
    Returns a list of encodings and a corresponding list of names.
    """
    files = os.listdir(folder_path)
    images = [file for file in files if file.endswith(('.png', '.jpg', '.jpeg'))]
    
    encodings = []
    for image in images:
        known_image = face_recognition.load_image_file(os.path.join(folder_path, image))
        # Check if there are any face encodings in the image
        face_encodings = face_recognition.face_encodings(known_image)
        if len(face_encodings) > 0:
            encodings.append(face_encodings[0])
    
    names = [name] * len(encodings)
    return encodings, names

def identify_face(input_data=None, input_path=None):

    known_encodings = []
    known_names = []

    result = ""

    # Load stored known faces if the JSON file exists
    if os.path.exists("known_faces.json"):
        with open("known_faces.json", "r") as json_file:
            stored_data = json.load(json_file)
        known_encodings = [np.array(enc) for enc in stored_data["encodings"]]
        known_names = stored_data["names"]
    else:
        # If no stored data, generate and store it from the specified directories
        david_folder = module_dir + "/faces/David/"
        david_encodings, david_names = load_encodings_from_folder("David", david_folder)
        
        kevin_folder = module_dir + "/faces/Kevin/"
        kevin_encodings, kevin_names = load_encodings_from_folder("Kevin", kevin_folder)

        known_encodings = david_encodings + kevin_encodings
        known_names = david_names + kevin_names

        data_to_store = {
            "encodings": [enc.tolist() for enc in known_encodings],
            "names": known_names
        }

        with open("known_faces.json", "w") as json_file:
            json.dump(data_to_store, json_file)

    # Load the image based on input
    if input_data is not None:
        unknown_image = input_data
    elif input_path is not None:
        unknown_image = face_recognition.load_image_file(input_path)
    else:
        raise ValueError("You must provide either input_data or input_path.")

    # Extract encodings from the image
    unknown_encodings = face_recognition.face_encodings(unknown_image)

    # Compare against known faces
    if len(unknown_encodings) == 0:
        result = "No faces found."
    else:
        for unknown_encoding in unknown_encodings:
            results = face_recognition.compare_faces(known_encodings, unknown_encoding)
            distances = face_recognition.face_distance(known_encodings, unknown_encoding)

            if True in results:
                best_match_index = results.index(True)
                name = known_names[best_match_index]
                distance = distances[best_match_index]

                if distance < 0.4:
                    result = f"{name}"
                else:
                    result = "Not recognized"
            else:
                result = "Not recognized"
    # print(result)
    return result

if __name__ == "__main__":

    print( "test face result: " + identify_face(None, module_dir + "/test_face.png"))