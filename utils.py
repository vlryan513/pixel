import re
from pathlib import Path
import requests
from deepface import DeepFace
import cv2
import os
from datetime import datetime

def extract_datetime(image):
    name = image['filename']
    ts = name.replace('esp32_image_', '').replace('.jpg', '')  # '20250501_140409'
    dt_str = ts[:8] + ts[9:]  # '20250501140409'
    return datetime.strptime(dt_str, '%Y%m%d%H%M%S')

# function to get actual file name
def get_filename(cd):
    """Extract filename from Content-Disposition header"""
    if not cd:
        return None
    fname = re.findall('filename="(.+)"', cd)
    if len(fname) == 0:
        return None
    return fname[0]


# update photos
def update_images(url):
    # get images from google apps script
    response = requests.get(url)

    image_urls = response.json()

    new_images = []

    folder = Path('static/images')

    # download images
    for url in image_urls:
        response = requests.get(url, allow_redirects=True)
        cd = response.headers.get('Content-Disposition')
        filename = get_filename(cd) or 'unknown.jpg'

        # with open(f"images/{filename}", "wb") as f:
        #     f.write(response.content)

        file_path = folder / filename

        if not file_path.exists():
            
            new_images.append(filename)
            with open(f"static/images/{filename}", "wb") as f:
                f.write(response.content)

    return(new_images)

def update_sort(face_db,url):
    # sort only new images

    response = requests.get(url)

    image_urls = response.json()

    folder = Path('static/images')

    # download images
    for url in image_urls:
        response = requests.get(url, allow_redirects=True)
        cd = response.headers.get('Content-Disposition')
        filename = get_filename(cd) or 'unknown.jpg'

        # with open(f"images/{filename}", "wb") as f:
        #     f.write(response.content)

        file_path = folder / filename

        if not file_path.exists():
            
            with open(f"static/images/{filename}", "wb") as f:
                f.write(response.content)

    image_folder = "static/images"

    all_files = os.listdir('static/images')

    # check for images not yet sorted
    used_files = set()
    for images in face_db.values():
        used_files.update(images)
    new_images = []
    for f in all_files:
        if f not in used_files:
            new_images.append(f)

    # iterate over new images
    for image_file in new_images:
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

    return(face_db)
