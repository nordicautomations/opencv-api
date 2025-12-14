import io, base64, math, re
from flask import Flask, request, jsonify
import numpy as np
import cv2
from pdf2image import convert_from_bytes
from PIL import Image
import pytesseract
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union

app = Flask(__name__)

# -------------------------
# Utils
# -------------------------

def b64_to_images(b64, dpi=300):
    raw = base64.b64decode(b64)
    if raw[:4] == b'%PDF':
        return convert_from_bytes(raw, dpi=dpi)
    return [Image.open(io.BytesIO(raw)).convert("RGB")]

def pil_to_cv(img):
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def polygon_area(points):
    if len(points) < 3:
        return 0.0
    x = np.array([p[0] for p in points])
    y = np.array([p[1] for p in points])
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

# -------------------------
# OCR
# -------------------------

def extract_text_boxes(cv_img, min_conf=40):
    pil = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
    data = pytesseract.image_to_data(pil, output_type=pytesseract.Output.DICT)
    boxes = []
    for i in range(len(data["text"])):
        txt = data["text"][i].strip()
        conf = int(data["conf"][i]) if data["conf"][i].isdigit() else -1
        if conf < min_conf or not txt:
            continue
        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        boxes.append({
            "text": txt,
            "confidence": round(conf / 100.0, 2),
            "bbox": [x, y, w, h],
            "centroid": [x + w // 2, y + h // 2]
        })
    return boxes

def mask_text_regions(cv_img, text_boxes):
    img = cv_img.copy()
    for t in text_boxes:
        x,y,w,h = t["bbox"]
        pad = int(max(2, min(w,h)*0.15))
        cv2.rectangle(
            img,
            (max(0,x-pad), max(0,y-pad)),
            (min(img.shape[1],x+w+pad), min(img.shape[0],y+h+pad)),
            (255,255,255),
            -1
        )
    return img

# -------------------------
# Walls + Rooms
# -------------------------

def detect_walls(cv_img):
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    dil = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT,(5,5)), 2)
    lines = cv2.HoughLinesP(dil, 1, math.pi/180, 120, minLineLength=60, maxLineGap=25)
    line_mask = np.zeros_like(gray)
    if lines is not None:
        for x1,y1,x2,y2 in lines.reshape(-1,4):
            cv2.line(line_mask, (x1,y1), (x2,y2), 255, 8)
    combined = cv2.bitwise_or(dil, line_mask)
    closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_RECT,(15,15)), 2)
    _, wall_mask = cv2.threshold(closed, 50, 255, cv2.THRESH_BINARY)
    return wall_mask

def extract_rooms(wall_mask, min_area_px=2500):
    inv = cv2.bitwise_not(wall_mask)
    contours, _ = cv2.findContours(inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rooms = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area_px:
            continue
        eps = 0.007 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, eps, True)
        poly = approx.reshape(-1,2).tolist()
        if len(poly) < 3:
            continue
        rooms.append(poly)
    return rooms

# -------------------------
# Room naming (NIVÅ 3)
# -------------------------

def assign_room_name(room_poly, text_boxes):
    poly = Polygon(room_poly)
    candidates = []
    for t in text_boxes:
        if poly.contains(Point(t["centroid"])):
            candidates.append(t)
    if not candidates:
        return {"value":"Ukjent","source":"none","confidence":0.0}
    best = max(candidates, key=lambda x: x["confidence"])
    return {
        "value": best["text"],
        "source": "ocr",
        "confidence": best["confidence"]
    }

# -------------------------
# API
# -------------------------

@app.route("/health")
def health():
    return jsonify({"status":"ok"})

@app.route("/process", methods=["POST"])
def process():
    data = request.get_json(force=True)
    if "image" not in data:
        return jsonify({"error":"missing image"}), 400

    pxpm = float(data["px_per_meter"]) if "px_per_meter" in data else None
    images = b64_to_images(data["image"])
    results = []

    for page, img in enumerate(images, start=1):
        cv_img = pil_to_cv(img)

        text_boxes = extract_text_boxes(cv_img)
        img_no_text = mask_text_regions(cv_img, text_boxes)
        walls = detect_walls(img_no_text)
        rooms = extract_rooms(walls)

        for i, poly in enumerate(rooms):
            area_px = polygon_area(poly)
            area_m2 = round(area_px / (pxpm**2), 3) if pxpm else 0.0
            name = assign_room_name(poly, text_boxes)

            results.append({
                "page": page,
                "room_id": f"P{page}_R{i+1}",
                "area_m2": area_m2,
                "area_px": round(area_px,2),
                "name": name,
                "polygon_px": poly
            })

    return jsonify({
        "status":"ok",
        "rooms_detected": len(results),
        "rooms": results
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
