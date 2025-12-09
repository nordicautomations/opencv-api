from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64

app = Flask(__name__)

@app.route("/process", methods=["POST"])
def process():
    data = request.json
    img_b64 = data["image"]
    img_data = base64.b64decode(img_b64)
    npimg = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"error": "Invalid image"}), 400

    h, w = img.shape[:2]

    return jsonify({
        "status": "ok",
        "width": w,
        "height": h
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
