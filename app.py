from flask import Flask, render_template, url_for, redirect
import requests
import os
import re
import shutil
from deepface import DeepFace
import cv2
from utils import get_filename, update_images, update_sort, extract_datetime
from pathlib import Path
from datetime import datetime

app = Flask(__name__)

# global variables
web_app_url = "https://script.google.com/macros/s/AKfycbw2KM-GIc6C7mTgzi7TRHD8AgMC4O5yFYrvp4h5Tzk_BjTTYsJdJw036nqDaJjhoCD7/exec"
# dictionary for detected faces
face_db = {}  # { "person_X": ["image1.jpg", "image2.jpg"] }

@app.route("/")
def home():
    folder = Path('static/images')
    if folder.exists():
        image_folder = os.path.join('static', 'images')
        image_names = os.listdir(image_folder)
        image_info = []

        for name in image_names:
            if name.endswith(".jpg"):  # optional filter
                image_info.append({
                    'url': url_for('static', filename=f'images/{name}'),
                    'filename': name
                })

        image_info.sort(key=extract_datetime, reverse=True)

        return render_template('home_with_images.html', images=image_info)
    return render_template('home.html')

@app.route('/run-download', methods=['GET'])
def download():

    # get images from google apps script
    response = requests.get(web_app_url)
    # response.json()

    image_urls = response.json()

    # create folder to store images
    os.makedirs("static/images", exist_ok=True)

    # download images
    for url in image_urls:
        response = requests.get(url, allow_redirects=True)
        cd = response.headers.get('Content-Disposition')
        filename = get_filename(cd) or 'unknown.jpg'

        with open(f"static/images/{filename}", "wb") as f:
            f.write(response.content)

        # print(f"Saved file as: {filename}")

    # image_folder = os.path.join('static', 'images')
    # image_names = os.listdir(image_folder)
    # image_paths = [url_for('static', filename=f'images/{name}') for name in image_names]

    # return render_template('downloads.html',images=image_paths)
    return redirect(url_for('home'))

@app.route('/files')
def list_files():
    files = os.listdir('static/images')  # Get list of files in the 'images/' folder
    return render_template('files.html', files=files)

@app.route('/run-sort',methods=['GET'])
def run_sort():

    global face_db

    # path to images
    image_folder = "static/images"

    # list all images
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png','.JPG'))]

    # if image file is already in the face_db, take it out of the list
    image_files = [img for img in image_files if img not in face_db]

    # item for images with no faces
    face_db["no_faces"] = []

    # iterate over all images
    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)

        # detect faces
        # use retinaface for better accuracy
        # only take faces with high confidence
        try:
            faces = DeepFace.extract_faces(image_path,detector_backend="retinaface",enforce_detection=False)
            faces = [face for face in faces if face.get("confidence", 1.0) > 0.9]

            if not faces:  # no faces found
                face_db["no_faces"].append(image_file)
                continue

            for face in faces:
                face_img = face["face"]

                # compare with known faces
                matched_person = None
                for person, saved_images in face_db.items():
                    if person == "no_faces":  # skip "no_faces" category
                        continue

                    reference_image = os.path.join(image_folder, saved_images[0])

                    # compare face embeddings
                    result = DeepFace.verify(image_path, reference_image, enforce_detection=False)

                    if result["verified"]:  # face matches
                        matched_person = person
                        break

                if matched_person:
                    face_db[matched_person].append(image_file)
                else:
                    new_person = f"person_{len(face_db)}"
                    face_db[new_person] = [image_file]

        except Exception as e:
            print(f"Skipping {image_file}: {e}")

    # remove duplicates
    for person in face_db:
        face_db[person] = list(set(face_db[person]))

    return render_template('sorted.html', data=face_db)

@app.route('/run-update', methods=['POST'])
def run_update():
    new = update_images(web_app_url)
    return redirect(url_for('home'))

@app.route('/run-update-sort', methods=['POST'])
def run_update_sort():
    global face_db
    updated_faces = update_sort(face_db,web_app_url)
    face_db = updated_faces
    return redirect(url_for('sorted'))

@app.route('/sorted')
def sorted():
    return render_template('sorted.html', data=face_db)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)