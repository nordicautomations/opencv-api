from flask import Flask, request, jsonify
import base64, cv2, numpy as np, math, re

app = Flask(__name__)

def decode_image_from_base64(b64):
    img_bytes = base64.b64decode(b64)
    arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img

def detect_walls(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    th = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,21,10)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)
    # Hough lines for strong straight walls
    edges = cv2.Canny(closed,50,150,apertureSize=3)
    lines = cv2.HoughLinesP(edges,1,math.pi/180,threshold=80,minLineLength=60,maxLineGap=10)
    wall_mask = np.zeros_like(gray)
    if lines is not None:
        for x1,y1,x2,y2 in lines.reshape(-1,4):
            cv2.line(wall_mask, (x1,y1), (x2,y2), 255, 7)
    # Combine structural mask
    wall_mask = cv2.dilate(wall_mask, kernel, iterations=2)
    wall_mask = cv2.bitwise_or(wall_mask, closed)
    wall_mask = cv2.erode(wall_mask, kernel, iterations=1)
    return wall_mask

def find_closed_regions(wall_mask):
    # invert so rooms are white
    inv = cv2.bitwise_not(wall_mask)
    # remove small specks
    inv = cv2.morphologyEx(inv, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)), iterations=1)
    contours, hierarchy = cv2.findContours(inv, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    regions = []
    if contours is None:
        return regions
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area < 2000:  # noise filter, tweak if needed
            continue
        # only external parents (or include holes via hierarchy)
        # keep contour if it's a leaf or parent depending on hierarchy
        # approx polygon
        eps = 0.01 * cv2.arcLength(cnt, True)
        poly = cv2.approxPolyDP(cnt, eps, True)
        poly2 = poly.reshape(-1,2).tolist()
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            cx,cy = 0,0
        else:
            cx = int(M["m10"]/M["m00"]); cy = int(M["m01"]/M["m00"])
        regions.append({"polygon": poly2, "area_px": float(area), "centroid":[cx,cy]})
    return regions

def polygon_area(px_poly):
    # shoelace
    x = [p[0] for p in px_poly]; y = [p[1] for p in px_poly]
    n = len(px_poly)
    area = 0.0
    for i in range(n):
        j = (i+1)%n
        area += x[i]*y[j] - x[j]*y[i]
    return abs(area)/2.0

def parse_scale_text(scale_text):
    # accepts "1:100" or "100" (meaning px_per_meter unknown) or "px_per_meter=150"
    if not scale_text: return None
    m = re.search(r'1\s*:\s*([0-9]+)', scale_text)
    if m:
        denom = float(m.group(1))
        return {"scale_type":"ratio","ratio":denom}
    m2 = re.search(r'px_per_meter\s*=\s*([0-9]+)', scale_text)
    if m2:
        return {"scale_type":"pxpm","px_per_meter":float(m2.group(1))}
    try:
        val = float(scale_text)
        return {"scale_type":"pxpm","px_per_meter":val}
    except:
        return None

@app.route("/process", methods=["POST"])
@app.route("/p", methods=["POST"])
def process():
    try:
        payload = request.get_json(force=True)
        if not payload or "image" not in payload:
            return jsonify({"error":"missing 'image' field"}), 400
        img = decode_image_from_base64(payload["image"])
        if img is None:
            return jsonify({"error":"failed to decode image"}), 400

        # optional user-supplied calibration
        pxpm = None
        area_method = "estimate"
        if "px_per_meter" in payload:
            try:
                pxpm = float(payload["px_per_meter"])
                area_method = "pxpm"
            except: pxpm=None
        elif "scale_text" in payload:
            s = parse_scale_text(payload.get("scale_text"))
            if s and s.get("scale_type")=="pxpm":
                pxpm = s.get("px_per_meter")
                area_method = "pxpm"

        wall_mask = detect_walls(img)
        regions = find_closed_regions(wall_mask)

        rooms=[]
        for r in regions:
            poly = r["polygon"]
            # clean polygon: ensure clockwise, remove duplicates
            if len(poly) < 3: continue
            area_px = polygon_area(poly)
            if pxpm:
                area_m2 = round(area_px / (pxpm**2), 3)
            else:
                area_m2 = 0.0
            rooms.append({
                "polygon_px": poly,
                "area_px": round(area_px,2),
                "area_m2": area_m2,
                "area_method": area_method,
                "centroid": r["centroid"]
            })

        # fallback estimate if no rooms found: try contour detection external
        if len(rooms)==0:
            # use previous simpler detect_rooms-like fallback
            edges = cv2.Canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),50,150)
            cnts,_ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                a=cv2.contourArea(c)
                if a<2000: continue
                poly = cv2.approxPolyDP(c,0.02*cv2.arcLength(c,True),True).reshape(-1,2).tolist()
                rooms.append({"polygon_px":poly,"area_px":float(a),"area_m2":0.0,"area_method":"estimate","centroid":[int(cv2.moments(c)['m10']/a) if a else 0,int(cv2.moments(c)['m01']/a) if a else 0]})

        return jsonify({"status":"ok","rooms_detected":len(rooms),"rooms":rooms})

    except Exception as e:
        return jsonify({"error":str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
