from flask import Flask, request, jsonify
import base64
import cv2
import numpy as np

app = Flask(__name__)

@app.route("/process", methods=["POST"])
@app.route("/p", methods=["POST"])
def process():
    try:
        data = request.get_json()

        if not data or "image" not in data:
            return jsonify({"error": "missing 'image' field"}), 400

        # Decode base64 → bytes
        img_bytes = base64.b64decode(data["image"])

        # Bytes → numpy array
        img_array = np.frombuffer(img_bytes, np.uint8)

        # Decode to OpenCV image
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"error": "Failed to decode image"}), 400

        # SUCCESS
        return jsonify({
            "status": "ok",
            "shape": img.shape,
            "message": "Image received and decoded successfully"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
