import argparse
import io
import math

from PIL import Image
import datetime
from gtts import gTTS
import pygame
import torch
import cv2
import numpy as np
import tensorflow as tf
from re import DEBUG, sub
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    send_file,
    url_for,
    Response,
)
from werkzeug.utils import secure_filename, send_from_directory
import os
import subprocess
from subprocess import Popen
import re
import requests
import shutil
import time
import glob


from ultralytics import YOLO


####### flick8r

import os
import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt
import string
import json
from pickle import load, dump
from nltk.translate.bleu_score import corpus_bleu

from ultralytics import YOLO
import cv2
import cvzone
import math
import time
from gtts import gTTS
import pygame
from googletrans import Translator

"""
1. Extract image features using pre-trained CNN.
"""


def feature_extractions(directory):
    """
    Input: directory of images
    Return: A dictionary of features extracted by VGG-16, size 4096.
    """

    model = tf.keras.applications.vgg16.VGG16()
    model = keras.models.Model(inputs=model.input, outputs=model.layers[-2].output)  # Remove the final layer

    features = {}
    for f in os.listdir(directory):
        filename = directory + "/" + f
        identifier = f.split('.')[0]

        image = keras.preprocessing.image.load_img(filename, target_size=(224, 224))
        arr = keras.preprocessing.image.img_to_array(image, dtype=np.float32)
        arr = arr.reshape((1, arr.shape[0], arr.shape[1], arr.shape[2]))
        arr = keras.applications.vgg16.preprocess_input(arr)

        feature = model.predict(arr, verbose=0)
        features[identifier] = feature

        print("feature extraction: {}".format(f))
    return (features)


# =============================================================================
#
# if __name__ == "__main__":
#     features = feature_extractions("Flickr8k_Dataset")
#
#     #save features for future use.
#     with open("features.pkl", "wb") as f:
#        dump(features, f)
# =============================================================================


def sample_caption(model, tokenizer, max_length, vocab_size, feature):
    """
    Input: model, photo feature: shape=[1,4096]
    Return: A generated caption of that photo feature. Remove the startseq and endseq token.
    """

    caption = "<startseq>"
    while 1:
        # Prepare input to model
        encoded = tokenizer.texts_to_sequences([caption])[0]
        padded = keras.preprocessing.sequence.pad_sequences([encoded], maxlen=max_length, padding='pre')[0]
        padded = padded.reshape((1, max_length))

        pred_Y = model.predict([feature, padded])[0, -1, :]
        next_word = tokenizer.index_word[pred_Y.argmax()]

        # Update caption
        caption = caption + ' ' + next_word

        # Terminate condition: caption length reaches maximum / reach endseq
        if next_word == '<endseq>' or len(caption.split()) >= max_length:
            break

    # Remove the (startseq, endseq)
    caption = caption.replace('<startseq> ', '')
    caption = caption.replace(' <endseq>', '')

    return (caption)


##############


def caption_al():
    import os
    import keras
    from keras.preprocessing.text import tokenizer_from_json
    import numpy as np
    import matplotlib.pyplot as plt
    import json
    from pickle import load, dump

    # Load tokenizer
    with open('tokenizer.json', 'r') as f:
        tokenizer_json = json.load(f)
    tokenizer = tokenizer_from_json(tokenizer_json)

    model = keras.models.load_model("./sample_model.h5")  # Load model
    vocab_size = tokenizer.num_words  # The number of vocabulary
    max_length = 37  # Maximum length of caption sequence

    # sampling
    features = feature_extractions("./sample_images")

    for i, filename in enumerate(features.keys()):
        plt.figure(i + 1)
        caption = sample_caption(model, tokenizer, max_length, vocab_size, features[filename])

        img = keras.preprocessing.image.load_img("./sample_images/{fn}.jpg".format(fn=filename))

    return caption

def translate_text(text, target_language='tr'):
    translator = Translator()
    translated_text = translator.translate(text, dest=target_language)
    return translated_text.text



app = Flask(__name__)

pygame.mixer.init()
@app.route("/")
def hello_world():
    # return render_template("index.html")
    if "image_path" in request.args:
        image_path = request.args["image_path"]
        return render_template("index.html", image_path=image_path)
    return render_template("index.html")

@app.route("/", methods=["GET", "POST"])
def predict_img():
    if request.method == "POST":
        detection_type = request.form['detectionType']
        print(detection_type,"YAKUP DETECTION TYPE")
        if 'file' in request.files:
            f = request.files['file']
            basepath = os.path.dirname(__file__)
            filepath = os.path.join(basepath, 'uploads', f.filename)
            print("upload folder is ", filepath)
            f.save(filepath)
            predict_img.imgpath = f.filename
            print("printing predict_img :::::: ", predict_img)

            file_extension = f.filename.rsplit('.', 1)[1].lower()

            if detection_type == 'objectDetection':

                if file_extension == 'jpg':
                    # Handle image upload
                    img = cv2.imread(filepath)
                    model = YOLO('best.pt')
                    detections = model(img, save=True)

                    # Find the latest subdirectory in the 'runs/detect' folder
                    folder_path = os.path.join(basepath, 'runs', 'detect')
                    subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
                    latest_subfolder = max(subfolders, key=lambda x: os.path.getctime(os.path.join(folder_path, x)))

                    # Construct the relative path to the detected image file
                    static_folder = os.path.join(basepath, 'static', 'assets')
                    relative_image_path = os.path.relpath(os.path.join(folder_path, latest_subfolder, f.filename), static_folder)
                    image_path = os.path.join(folder_path, latest_subfolder, f.filename)
                    print("Relative image path:", relative_image_path)  # Print the relative_image_path for debugging

                    # Process detections and generate TTS
                    largest_bbox_area = 0
                    largest_bbox_object = ""
                    screen_center_x = img.shape[1] // 2

                    for result in detections:
                        boxes = result.boxes
                        for box in boxes:
                            conf = math.ceil((box.conf[0] * 100)) / 100
                            if conf < 0.5:
                                continue

                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            w, h = x2 - x1, y2 - y1
                            area = w * h
                            if area > largest_bbox_area:
                                largest_bbox_area = area
                                largest_bbox_object = result.names[int(box.cls[0])]
                                cv2.imwrite(f'./sample_images/image1.jpg', img)

                            if largest_bbox_object != "" and x1 > screen_center_x and x1 - 20 > screen_center_x:
                                yon = "Sağ tarafta"
                            elif largest_bbox_object != "" and x2 < screen_center_x and x2 + 20 < screen_center_x:
                                yon = "Sol tarafta"
                            else:
                                if largest_bbox_object != "":
                                    yon = "Ön tarafta"

                    if largest_bbox_object:
                        text = f'{yon} {largest_bbox_object} var'
                        speech = gTTS(text=text, lang="tr", slow=False)
                        speech.save('output.mp3')


                    return render_template('index.html', image_path=relative_image_path, media_type='image')

                elif file_extension == "mp4":
                    video_path = filepath  # replace with your video path
                    cap = cv2.VideoCapture(video_path)

                    # get video dimensions
                    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                    # Define the codec and create VideoWriter object
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    out = cv2.VideoWriter("output.mp4", fourcc, 30.0, (frame_width, frame_height))

                    # initialize the YOLOv8 model here
                    model = YOLO("best.pt")
                    start_time = time.time()
                    sayac = 0

                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break

                        results = model(frame, save=True)
                        largest_bbox_area = 0
                        largest_bbox_object = ""
                        screen_center_x = frame.shape[1] // 2

                        for r in results:
                            boxes = r.boxes
                            for box in boxes:
                                conf = math.ceil((box.conf[0] * 100)) / 100
                                if conf < 0.5:
                                    continue

                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                w, h = x2 - x1, y2 - y1
                                area = w * h
                                if area > largest_bbox_area:
                                    largest_bbox_area = area
                                    largest_bbox_object = r.names[int(box.cls[0])]
                                    cv2.imwrite(f'./sample_images/image1.jpg', frame)

                                if largest_bbox_object != "" and x1 > screen_center_x and x1 - 20 > screen_center_x:
                                    yon = "Sağ tarafta"
                                elif largest_bbox_object != "" and x2 < screen_center_x and x2 + 20 < screen_center_x:
                                    yon = "Sol tarafta"
                                else:
                                    if largest_bbox_object != "":
                                        yon = "Ön tarafta"

                        elapsed_time = time.time() - start_time
                        if elapsed_time >= 5 and largest_bbox_object:
                            text = f'{yon} {largest_bbox_object} var'
                            speech = gTTS(text=text, lang="tr", slow=False)
                            file_path = f'static/assets/output{sayac}.mp3'
                            speech.save(file_path)
                            sayac += 1
                            start_time = time.time()

                        res_plotted = results[0].plot()
                        out.write(res_plotted)

                    return render_template('index.html', video_path='output.mp4', media_type='video',sayac=sayac-1,detectionType="objectDetection")

            if detection_type == 'motionDetection':
                if file_extension == 'jpg':

                    img = cv2.imread(filepath)
                    model = YOLO('best.pt')
                    detections = model(img, save=True)


                    folder_path = os.path.join(basepath, 'runs', 'detect')
                    subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
                    latest_subfolder = max(subfolders, key=lambda x: os.path.getctime(os.path.join(folder_path, x)))


                    static_folder = os.path.join(basepath, 'static', 'assets')
                    relative_image_path = os.path.relpath(os.path.join(folder_path, latest_subfolder, f.filename),
                                                          static_folder)
                    image_path = os.path.join(folder_path, latest_subfolder, f.filename)
                    print("Relative image path:", relative_image_path)


                    largest_bbox_area = 0
                    largest_bbox_object = ""
                    screen_center_x = img.shape[1] // 2

                    for result in detections:
                        boxes = result.boxes
                        for box in boxes:
                            conf = math.ceil((box.conf[0] * 100)) / 100
                            if conf < 0.5:
                                continue

                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            w, h = x2 - x1, y2 - y1
                            area = w * h
                            if area > largest_bbox_area:
                                largest_bbox_area = area
                                largest_bbox_object = result.names[int(box.cls[0])]
                                cv2.imwrite(f'./sample_images/image1.jpg', img)

                            if largest_bbox_object != "" and x1 > screen_center_x and x1 - 20 > screen_center_x:
                                yon = "Sağ tarafta"
                            elif largest_bbox_object != "" and x2 < screen_center_x and x2 + 20 < screen_center_x:
                                yon = "Sol tarafta"
                            else:
                                if largest_bbox_object != "":
                                    yon = "Ön tarafta"

                    if largest_bbox_object:
                        text2= yon + translate_text(caption_al(), target_language='tr')
                        speech = gTTS(text=text2, lang="tr", slow=False)
                        speech.save('static/assets/outputHareket.mp3')

                    return render_template('index.html', image_path=relative_image_path, media_type='image')

                elif file_extension == "mp4":
                    video_path = filepath
                    cap = cv2.VideoCapture(video_path)


                    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    out = cv2.VideoWriter("output.mp4", fourcc, 30.0, (frame_width, frame_height))


                    model = YOLO("yolov8n.pt")
                    start_time = time.time()
                    sayac = 0

                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break

                        results = model(frame, save=True)
                        largest_bbox_area = 0
                        largest_bbox_object = ""
                        screen_center_x = frame.shape[1] // 2

                        for r in results:
                            boxes = r.boxes
                            for box in boxes:
                                conf = math.ceil((box.conf[0] * 100)) / 100
                                if conf < 0.5:
                                    continue

                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                w, h = x2 - x1, y2 - y1
                                area = w * h
                                if area > largest_bbox_area:
                                    largest_bbox_area = area
                                    largest_bbox_object = r.names[int(box.cls[0])]
                                    cv2.imwrite(f'./sample_images/image1.jpg', frame)

                                if largest_bbox_object != "" and x1 > screen_center_x and x1 - 20 > screen_center_x:
                                    yon = "Sağ tarafta"
                                elif largest_bbox_object != "" and x2 < screen_center_x and x2 + 20 < screen_center_x:
                                    yon = "Sol tarafta"
                                else:
                                    if largest_bbox_object != "":
                                        yon = "Ön tarafta"

                        elapsed_time = time.time() - start_time
                        if elapsed_time >= 5 and largest_bbox_object:
                            try:
                                text2= yon + translate_text(caption_al(), target_language='tr')
                                speech = gTTS(text=text2, lang="tr", slow=False)
                                speech.save(f'static/assets/outputHareket{sayac}.mp3')
                                sayac += 1
                            except:
                                print("yakup hata logu")

                            start_time = time.time()



                        res_plotted = results[0].plot()
                        out.write(res_plotted)

                    return render_template('index.html', video_path='output.mp4', media_type='video', sayac=sayac,detectionType="motionDetection")

    return render_template("index.html", image_path="", media_type='image')

@app.route("/<path:filename>")
def display(filename):
    folder_path = "runs/detect"
    subfolders = [
        f
        for f in os.listdir(folder_path)
        if os.path.isdir(os.path.join(folder_path, f))
    ]
    latest_subfolder = max(
        subfolders, key=lambda x: os.path.getctime(os.path.join(folder_path, x))
    )
    directory = os.path.join(folder_path, latest_subfolder)
    print("printing directory: ", directory)
    files = os.listdir(directory)
    latest_file = files[0]

    print(latest_file)

    image_path = os.path.join(directory, latest_file)

    file_extension = latest_file.rsplit(".", 1)[1].lower()

    if file_extension == "jpg":
        return send_file(image_path, mimetype="image/jpeg")
    elif file_extension == "mp4":
        return send_file(image_path, mimetype="video/mp4")
    else:
        return "Invalid file format"


def get_frame():
    folder_path = os.getcwd()
    mp4_files = "output.mp4"
    print("files being read...")
    video = cv2.VideoCapture(mp4_files)
    while True:
        success, frame = video.read()
        if not success:
            print("file not being read")
            break
        else:
            ret, buffer = cv2.imencode(".jpg", frame)
            frame = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n\r\n"
        )
        time.sleep(0.1)




@app.route("/video_feed")
def video_feed():
    #folder_path = os.getcwd()
    #mp4_file = "static/assets/output.mp4"
    #video_path = os.path.join(folder_path, mp4_file)
    #return send_file(video_path, mimetype="video")
    return Response(get_frame(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/webcam_feed")
def webcam_feed():
    cap = cv2.VideoCapture(0) # 0 for camera

    def generate():
        while True:
            success, frame = cap.read()
            if not success:
                break


            img = Image.fromarray(frame)
            model = YOLO("best.pt")
            results = model(img, save=True)

            # Plot the detected objects on the frame
            res_plotted = results[0].plot()
            img_BGR = cv2.cvtColor(res_plotted, cv2.COLOR_RGB2BGR)

            # Convert the frame to JPEG format for streaming
            ret, buffer = cv2.imencode(".jpg", img_BGR)
            frame = buffer.tobytes()

            yield (
                b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n\r\n"
            )

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask app exposing yolov8 models")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()
    model = YOLO("yolov8n.pt")
    app.run(host="0.0.0.0", port=args.port, debug=True)
