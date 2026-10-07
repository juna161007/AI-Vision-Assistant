from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np

app = Flask(__name__)

# Load YOLO model
model = YOLO("yolo11n.pt")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/detect", methods=["POST"])
def detect():

    file = request.files.get("image")

    if file is None:
        return jsonify({"error": "No image received"})


    image_bytes = np.frombuffer(
        file.read(),
        np.uint8
    )


    image = cv2.imdecode(
        image_bytes,
        cv2.IMREAD_COLOR
    )


    if image is None:
        return jsonify({"error": "Invalid image"})


    results = model(
        image,
        verbose=False
    )


    detected_objects = []


    for result in results:

        for box in result.boxes:

            class_id = int(
                box.cls[0]
            )


            confidence = float(
                box.conf[0]
            )


            if confidence >= 0.40:

                object_name = model.names[class_id]


                detected_objects.append({

                    "name": object_name,

                    "confidence":
                        round(
                            confidence * 100,
                            1
                        )

                })


    return jsonify({

        "objects":
            detected_objects

    })


if __name__ == "__main__":

    app.run(
        debug=True
    )