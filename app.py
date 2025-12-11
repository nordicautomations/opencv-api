# app.py
import io, base64, math, re, os, tempfile
from flask import Flask, request, jsonify
import numpy as np
import cv2
from pdf2image import convert_from_bytes
from PIL import Image
import pytesseract
from shapely.geometry import Polygon
from shapely.ops import unary_union

app = Flask(__name__)

# -------------------------
# Utilities
# -------------------------
def b64_to_images(b64, content_type_hint=None, dpi=300):
    """Accept either a PDF (bytes) or image (jpg/png). Return list of PIL Images."""
    raw = base64.b64decode(b64)
    # Heuristic: PDF begins with %PDF
    if raw[:4] == b'%PDF':
        imgs = convert_from_bytes(raw, dpi=dpi)
        return imgs
    else:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        return [img]

def pil_to_cv(img_pil):
    arr = np.array(img_pil)
    # PIL uses RGB, OpenCV uses BGR
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

def cv_to_b64jpg(cv_img, quality=90):
    _, enc = cv2.imencode('.jpg', cv_img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return base64.b64encode(enc.tobytes()).decode('ascii')

def parse_scale_text(scale_text):
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

def polygon_area_shoelace(points):
    if len(points) < 3: return 0.0
    x = [p[0] for p in points]; y = [p[1] for p in points]
    n = len(points)
    area = 0.0
    for i in range(n):
        j = (i+1)%n
        area += x[i]*y[j] - x[j]*y[i]
    return abs(area)/2.0

# -------------------------
# Preprocessing and masks
# -------------------------
def mask_text_regions(cv_img, oem=3, psm=3):
    """Use pytesseract to detect text boxes and mask them (fill white)."""
    pil = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
    # Use image_to_data for boxes
    data = pytesseract.image_to_data(pil, output_type=pytesseract.Output.DICT)
    img_masked = cv_img.copy()
    n_boxes = len(data['level'])
    for i in range(n_boxes):
        conf = int(data['conf'][i]) if data['conf'][i].isdigit() else -1
        if conf < 20:  # ignore poor detections
            continue
        x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
        # Expand a bit
        pad = int(max(2, min(w,h)*0.15))
        x0 = max(0, x-pad); y0 = max(0, y-pad)
        x1 = min(img_masked.shape[1]-1, x+w+pad); y1 = min(img_masked.shape[0]-1, y+h+pad)
        cv2.rectangle(img_masked, (x0,y0), (x1,y1), (255,255,255), thickness=-1)
    return img_masked

def detect_wall_mask(cv_img, debug=False):
    """Return binary mask of walls (255 = walls)."""
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    # Smooth and enhance edges
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    edges = cv2.Canny(blur, 50, 150)
    # Dilate to close gaps
    kern1 = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    dil = cv2.dilate(edges, kern1, iterations=2)
    # Hough line reinforcement (long straight walls)
    lines = cv2.HoughLinesP(dil, 1, math.pi/180, threshold=120, minLineLength=max(60, int(min(cv_img.shape[:2])*0.02)), maxLineGap=25)
    line_mask = np.zeros_like(gray)
    if lines is not None:
        for x1,y1,x2,y2 in lines.reshape(-1,4):
            cv2.line(line_mask, (x1,y1), (x2,y2), 255, 8)
    # Combine
    combined = cv2.bitwise_or(dil, line_mask)
    # Close large gaps and remove small specks
    bigk = cv2.getStructuringElement(cv2.MORPH_RECT, (15,15))
    closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, bigk, iterations=2)
    closed = cv2.dilate(closed, cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)), iterations=1)
    # Final wall mask: threshold
    _, wall_mask = cv2.threshold(closed, 50, 255, cv2.THRESH_BINARY)
    # Optional thin erosion to sharpen lines
    wall_mask = cv2.erode(wall_mask, cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)), iterations=1)
    if debug:
        return wall_mask, edges, combined
    return wall_mask

# -------------------------
# Room extraction
# -------------------------
def find_room_polygons_from_wall_mask(wall_mask, min_area_px=2500, debug=False):
    """
    Given binary wall_mask (255 walls, 0 background), invert and find closed regions.
    Apply filtering heuristics to remove furniture/text blobs.
    """
    # invert: rooms become foreground
    inv = cv2.bitwise_not(wall_mask)
    # open to remove thin artifacts
    inv = cv2.morphologyEx(inv, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)), iterations=1)
    # fill small holes inside potential rooms to avoid fragmentation
    inv = cv2.morphologyEx(inv, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT,(7,7)), iterations=2)

    contours, hierarchy = cv2.findContours(inv, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    if not contours:
        return polys
    for idx, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area < min_area_px:
            continue
        # bounding box and aspect
        x,y,w,h = cv2.boundingRect(cnt)
        aspect = w/h if h>0 else 0
        # approx polygon
        eps = 0.007 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, eps, True)
        pts = approx.reshape(-1,2).tolist()
        # compute solidity = area / convex_area
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull) if hull is not None else area
        solidity = float(area)/hull_area if hull_area>0 else 0
        # filter heuristics:
        # - reasonable aspect ratio (avoid very long thin artifacts)
        # - solidity not extremely low (to drop highly concave noise)
        # - area threshold (already)
        if (aspect > 10 or aspect < 0.1) and area < (min_area_px * 5):
            continue
        if solidity < 0.3:
            # highly concave -> likely noise, skip
            continue
        # accept
        polys.append({"poly": pts, "area_px": float(area), "bbox": [int(x),int(y),int(w),int(h)], "solidity": solidity})
    # Merge polygons that overlap heavily (adjacent fragments)
    final_polys = merge_close_polygons(polys)
    return final_polys

def merge_close_polygons(polys, iou_thresh=0.15):
    """Merge polygons that overlap significantly using shapely unions."""
    shapely_polys = []
    for p in polys:
        try:
            shp = Polygon(p["poly"]).convex_hull
            if not shp.is_valid or shp.area == 0:
                continue
            shapely_polys.append((shp, p))
        except Exception:
            continue
    if not shapely_polys:
        return []
    used = [False]*len(shapely_polys)
    merged = []
    for i,(shp_i, meta_i) in enumerate(shapely_polys):
        if used[i]: continue
        group = [shp_i]
        used[i]=True
        for j,(shp_j, meta_j) in enumerate(shapely_polys):
            if used[j]: continue
            inter = shp_i.intersection(shp_j).area
            union = shp_i.union(shp_j).area
            if union>0 and (inter/union) > iou_thresh:
                group.append(shp_j)
                used[j]=True
        if len(group)==1:
            merged_poly = shp_i
        else:
            merged_poly = unary_union(group)
        if merged_poly.geom_type == "Polygon":
            coords = list(map(lambda p:[int(p[0]),int(p[1])], merged_poly.exterior.coords[:-1]))
            merged.append({"poly": coords, "area_px": float(merged_poly.area)})
        else:
            # MultiPolygon -> take each
            for part in merged_poly:
                coords = list(map(lambda p:[int(p[0]),int(p[1])], part.exterior.coords[:-1]))
                merged.append({"poly": coords, "area_px": float(part.area)})
    return merged

# -------------------------
# Main process endpoint
# -------------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status":"ok"})

@app.route("/process", methods=["POST"])
@app.route("/p", methods=["POST"])
def process():
    try:
        payload = request.get_json(force=True)
        if not payload or "image" not in payload:
            return jsonify({"error":"missing 'image' field (base64)"}), 400
        images_pil = b64_to_images(payload["image"], dpi=300)
        # optional calibration
        pxpm = None
        area_method = "estimate"
        if "px_per_meter" in payload:
            try:
                pxpm = float(payload["px_per_meter"]); area_method="pxpm"
            except: pxpm=None
        elif "scale_text" in payload:
            s = parse_scale_text(payload.get("scale_text"))
            if s and s.get("scale_type")=="pxpm":
                pxpm = s.get("px_per_meter"); area_method="pxpm"

        all_rooms = []
        page_idx = 0
        for pil in images_pil:
            page_idx += 1
            cv_img = pil_to_cv(pil)
            # optional region-of-interest shrinking for massive images
            h, w = cv_img.shape[:2]
            if max(h,w) > 4000:
                # scale down for processing, but keep factor for px->m later
                scale = 4000.0 / max(h,w)
                cv_small = cv2.resize(cv_img, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                scale_factor = scale
            else:
                cv_small = cv_img.copy()
                scale_factor = 1.0

            # 1) Mask text (to avoid splitting rooms by printed labels)
            try:
                img_no_text = mask_text_regions(cv_small)
            except Exception:
                img_no_text = cv_small

            # 2) Detect walls
            wall_mask = detect_wall_mask(img_no_text)

            # 3) Extract polygons
            rooms = find_room_polygons_from_wall_mask(wall_mask, min_area_px=max(2000, int(0.0005*cv_small.shape[0]*cv_small.shape[1])))

            # 4) map back to original scale if needed and compute areas
            for r in rooms:
                poly = r["poly"]
                if scale_factor != 1.0:
                    inv_scale = 1.0/scale_factor
                    poly = [[int(p[0]*inv_scale), int(p[1]*inv_scale)] for p in poly]
                area_px = polygon_area_shoelace(poly)
                if pxpm:
                    area_m2 = round(area_px / (pxpm**2), 3)
                else:
                    area_m2 = 0.0
                # compute centroid
                try:
                    poly_np = np.array(poly)
                    M = cv2.moments(poly_np)
                    if M["m00"] != 0:
                        cx = int(M["m10"]/M["m00"]); cy = int(M["m01"]/M["m00"])
                    else:
                        cx,cy = 0,0
                except Exception:
                    cx,cy = 0,0
                all_rooms.append({
                    "page": page_idx,
                    "polygon_px": poly,
                    "area_px": round(area_px,2),
                    "area_m2": area_m2,
                    "area_method": area_method,
                    "centroid": [cx,cy]
                })
        return jsonify({"status":"ok","rooms_detected":len(all_rooms),"rooms": all_rooms})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
