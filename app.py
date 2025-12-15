import io, base64, math
from flask import Flask, request, jsonify
import numpy as np
import cv2
from PIL import Image

app = Flask(__name__)

# -------------------------
# Utils
# -------------------------

def b64_to_cv(b64):
    raw = base64.b64decode(b64)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def longest_wall_px(cv_img, orientation="horizontal"):
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi/180,
        threshold=120,
        minLineLength=200,
        maxLineGap=20
    )

    longest = 0
    if lines is None:
        return 0

    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)

        if orientation == "horizontal" and dx > dy * 2:
            longest = max(longest, dx)
        elif orientation == "vertical" and dy > dx * 2:
            longest = max(longest, dy)

    return longest

# -------------------------
# API
# -------------------------

@app.route("/process", methods=["POST"])
def process():
    data = request.get_json(force=True)

    if "image" not in data or "dimensions_summary" not in data:
        return jsonify({"error": "Missing image or dimensions_summary"}), 400

    img = b64_to_cv(data["image"])

    horiz_mm = data["dimensions_summary"].get("horizontal_mm")
    vert_mm  = data["dimensions_summary"].get("vertical_mm")

    if not horiz_mm or not vert_mm:
        return jsonify({"error": "Invalid dimensions_summary"}), 400

    horiz_px = longest_wall_px(img, "horizontal")
    vert_px  = longest_wall_px(img, "vertical")

    if horiz_px == 0 or vert_px == 0:
        return jsonify({"error": "Failed to detect scale lines"}), 400

    px_per_mm_h = horiz_px / horiz_mm
    px_per_mm_v = vert_px / vert_mm
    px_per_mm = (px_per_mm_h + px_per_mm_v) / 2

    px_per_meter = px_per_mm * 1000

    return jsonify({
        "status": "ok",
        "scale": {
            "px_per_mm": round(px_per_mm, 4),
            "px_per_meter": round(px_per_meter, 2)
        },
        "debug": {
            "horizontal_px": horiz_px,
            "vertical_px": vert_px
        }
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
