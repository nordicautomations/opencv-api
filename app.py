from flask import Flask, request, jsonify
import base64
import cv2
import numpy as np

app = Flask(__name__)


def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    # "Lukker" konturer så rom henger sammen
    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=2)
    edges = cv2.erode(edges, kernel, iterations=2)

    return edges


def detect_rooms(edge_img):
    contours, _ = cv2.findContours(edge_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rooms = []

    for c in contours:
        area = cv2.contourArea(c)
        if area < 5000:  # filtrer bort støy
            continue

        # polygon approx
        epsilon = 0.01 * cv2.arcLength(c, True)
        poly = cv2.approxPolyDP(c, epsilon, True)

        rooms.append({
            "polygon": poly.reshape(-1, 2).tolist(),
            "area_px": float(area)
        })

    return rooms


def convert_to_m2(area_px, px_per_meter=150):
    # Estimering — du kan endre etter kalibrering
    return round(area_px / (px_per_meter ** 2), 2)


@app.route("/process", methods=["POST"])
@app.route("/p", methods=["POST"])
def process():
    try:
        data = request.get_json()

        if not data or "image" not in data:
            return jsonify({"error": "missing 'image' field"}), 400

        img_bytes = base64.b64decode(data["image"])
        img_array = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"error": "failed to decode image"}), 400

        edges = preprocess_image(img)
        rooms = detect_rooms(edges)

        # legg på m²
        output = []
        for r in rooms:
            output.append({
                "polygon": r["polygon"],
                "area_pixels": r["area_px"],
                "area_m2_estimated": convert_to_m2(r["area_px"])
            })

        return jsonify({
            "status": "ok",
            "rooms_detected": len(output),
            "rooms": output
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
