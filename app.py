import io, base64, math
from flask import Flask, request, jsonify, send_file
import numpy as np
import cv2
from PIL import Image
from collections import defaultdict
import pytesseract
import pandas as pd
from datetime import datetime

app = Flask(__name__)

# -------------------------
# Enhanced Image Processing Utils
# -------------------------

def b64_to_cv(b64):
    """Convert base64 string to OpenCV image"""
    raw = base64.b64decode(b64)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def preprocess_floorplan(img):
    """Advanced preprocessing for architectural drawings"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
    binary = cv2.adaptiveThreshold(
        denoised, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 
        blockSize=15, 
        C=8
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    return binary, denoised

def detect_main_structure_lines(binary_img, orientation="horizontal"):
    """Enhanced line detection using multiple methods"""
    height, width = binary_img.shape
    min_length = (width if orientation == "horizontal" else height) * 0.3
    
    lines = cv2.HoughLinesP(
        binary_img, 
        rho=1, 
        theta=np.pi/180,
        threshold=int(min_length * 0.4),
        minLineLength=int(min_length),
        maxLineGap=30
    )
    
    if lines is None:
        return []
    
    valid_lines = []
    angle_tolerance = 15
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        angle = math.degrees(math.atan2(abs(y2 - y1), abs(x2 - x1)))
        
        if orientation == "horizontal":
            if angle < angle_tolerance or angle > (180 - angle_tolerance):
                valid_lines.append({
                    'length': length,
                    'coords': (x1, y1, x2, y2),
                    'angle': angle,
                    'midpoint_y': (y1 + y2) / 2
                })
        else:
            if abs(angle - 90) < angle_tolerance:
                valid_lines.append({
                    'length': length,
                    'coords': (x1, y1, x2, y2),
                    'angle': angle,
                    'midpoint_x': (x1 + x2) / 2
                })
    
    return valid_lines

def cluster_lines(lines, orientation="horizontal", tolerance=50):
    """Group parallel lines that likely represent the same wall"""
    if not lines:
        return []
    
    clusters = defaultdict(list)
    position_key = 'midpoint_y' if orientation == "horizontal" else 'midpoint_x'
    sorted_lines = sorted(lines, key=lambda x: x[position_key])
    
    cluster_id = 0
    for line in sorted_lines:
        if not clusters or abs(line[position_key] - 
                               clusters[cluster_id][-1][position_key]) > tolerance:
            cluster_id += 1
        clusters[cluster_id].append(line)
    
    representative_lines = []
    for cluster in clusters.values():
        longest = max(cluster, key=lambda x: x['length'])
        representative_lines.append(longest)
    
    return representative_lines

def find_building_dimensions(img, orientation="horizontal"):
    """Main function to find building dimensions"""
    binary, denoised = preprocess_floorplan(img)
    raw_lines = detect_main_structure_lines(binary, orientation)
    
    if not raw_lines:
        edges = cv2.Canny(denoised, 30, 100)
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        raw_lines = detect_main_structure_lines(edges, orientation)
    
    if not raw_lines:
        return 0, []
    
    clustered_lines = cluster_lines(raw_lines, orientation)
    clustered_lines.sort(key=lambda x: x['length'], reverse=True)
    longest_length = int(clustered_lines[0]['length'])
    
    return longest_length, clustered_lines[:5]

# -------------------------
# IMPROVED Room Detection
# -------------------------

def detect_rooms(img, min_area_px=2000, max_area_px=None):
    """
    IMPROVED: More aggressive room detection with multiple strategies
    """
    binary, _ = preprocess_floorplan(img)
    
    # Strategy 1: Standard inversion
    inverted = cv2.bitwise_not(binary)
    
    # Strategy 2: Less aggressive closing to preserve smaller rooms
    kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(inverted, cv2.MORPH_CLOSE, kernel_small, iterations=2)
    
    # Find contours with hierarchy
    contours, hierarchy = cv2.findContours(closed, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    
    if max_area_px is None:
        max_area_px = img.shape[0] * img.shape[1] * 0.8
    
    rooms = []
    
    # RELAXED filtering
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        
        # Lower threshold, broader range
        if min_area_px < area < max_area_px:
            x, y, w, h = cv2.boundingRect(contour)
            
            # More lenient aspect ratio (allow corridors)
            aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 0
            
            # Allow up to 15:1 ratio (corridors/hallways)
            if aspect_ratio < 15 and w > 20 and h > 20:
                rooms.append({
                    'contour': contour,
                    'area_px': area,
                    'bbox': (x, y, w, h),
                    'center': (x + w//2, y + h//2)
                })
    
    # Sort by area
    rooms.sort(key=lambda x: x['area_px'], reverse=True)
    
    print(f"DEBUG: Found {len(rooms)} rooms with min_area_px={min_area_px}")
    
    return rooms

def extract_text_from_region(img, bbox, padding=10):
    """
    IMPROVED: Safer OCR with fallback
    """
    try:
        x, y, w, h = bbox
        
        # Add padding safely
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(img.shape[1] - x, w + 2*padding)
        h = min(img.shape[0] - y, h + 2*padding)
        
        # Extract region
        roi = img[y:y+h, x:x+w]
        
        if roi.size == 0:
            return ""
        
        # Preprocess
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # OCR with fallback (try without Norwegian first)
        try:
            text = pytesseract.image_to_string(thresh, config='--oem 3 --psm 6')
        except:
            text = ""
        
        # Clean
        text = text.strip().replace('|', '').replace('_', ' ')
        text = ' '.join(text.split())
        
        return text if len(text) > 1 else ""
    
    except Exception as e:
        print(f"OCR Error: {e}")
        return ""

def assign_room_names(img, rooms):
    """
    IMPROVED: Better fallback naming
    """
    # Common Norwegian room types
    room_keywords = {
        'bad': 'Bad',
        'wc': 'WC',
        'sov': 'Soverom',
        'stue': 'Stue',
        'kjøkken': 'Kjøkken',
        'gang': 'Gang',
        'entre': 'Entré',
        'bod': 'Bod'
    }
    
    for idx, room in enumerate(rooms):
        x, y, w, h = room['bbox']
        
        # Try OCR
        text = extract_text_from_region(img, room['bbox'], padding=20)
        
        if text and len(text) > 1:
            # Check for known room types
            text_lower = text.lower()
            for keyword, full_name in room_keywords.items():
                if keyword in text_lower:
                    room['name'] = full_name
                    break
            else:
                room['name'] = text[:50]
        else:
            # Smart fallback based on size/position
            area_m2 = room.get('area_m2', 0)
            
            if area_m2 > 15:
                room['name'] = f"Stort rom {idx + 1}"
            elif area_m2 < 3:
                room['name'] = f"Lite rom/bod {idx + 1}"
            else:
                room['name'] = f"Rom {idx + 1}"
    
    return rooms

def calculate_room_areas(rooms, px_per_mm):
    """Calculate room areas in square meters"""
    for room in rooms:
        area_px = room['area_px']
        area_mm2 = area_px / (px_per_mm ** 2)
        area_m2 = area_mm2 / 1_000_000
        
        room['area_m2'] = round(area_m2, 2)
        
        _, _, w_px, h_px = room['bbox']
        room['width_m'] = round((w_px / px_per_mm) / 1000, 2)
        room['height_m'] = round((h_px / px_per_mm) / 1000, 2)
    
    return rooms

def create_excel_report(rooms, scale_info, metadata=None):
    """Create Excel file with room data"""
    data = []
    for i, room in enumerate(rooms, 1):
        data.append({
            'Rom Nr.': i,
            'Rom Navn': room['name'],
            'Areal (m²)': room['area_m2'],
            'Bredde (m)': room['width_m'],
            'Lengde (m)': room['height_m'],
            'Areal (px)': room['area_px'],
            'Posisjon X': room['center'][0],
            'Posisjon Y': room['center'][1]
        })
    
    df = pd.DataFrame(data)
    
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Romanalyse', index=False)
        
        summary_data = {
            'Metrikk': [
                'Totalt antall rom',
                'Totalt areal (m²)',
                'Gjennomsnittlig romareal (m²)',
                'Største rom (m²)',
                'Minste rom (m²)',
                'Målestokk (px/mm)',
                'Målestokk (px/m)',
                'Analysedato'
            ],
            'Verdi': [
                len(rooms),
                round(sum(r['area_m2'] for r in rooms), 2),
                round(sum(r['area_m2'] for r in rooms) / len(rooms), 2) if rooms else 0,
                max(r['area_m2'] for r in rooms) if rooms else 0,
                min(r['area_m2'] for r in rooms) if rooms else 0,
                scale_info.get('px_per_mm', 'N/A'),
                scale_info.get('px_per_meter', 'N/A'),
                datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            ]
        }
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Sammendrag', index=False)
        
        for sheet_name in writer.sheets:
            worksheet = writer.sheets[sheet_name]
            for column in worksheet.columns:
                max_length = 0
                column_letter = column[0].column_letter
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(cell.value)
                    except:
                        pass
                adjusted_width = min(max_length + 2, 50)
                worksheet.column_dimensions[column_letter].width = adjusted_width
    
    output.seek(0)
    return output

# -------------------------
# API Endpoints
# -------------------------

@app.route("/process", methods=["POST"])
def process():
    """Original scale detection endpoint"""
    try:
        data = request.get_json(force=True)
        
        if "image" not in data or "dimensions_summary" not in data:
            return jsonify({"error": "Missing image or dimensions_summary"}), 400
        
        img = b64_to_cv(data["image"])
        horiz_mm = data["dimensions_summary"].get("horizontal_mm")
        vert_mm = data["dimensions_summary"].get("vertical_mm")
        
        if not horiz_mm or not vert_mm:
            return jsonify({"error": "Invalid dimensions_summary"}), 400
        
        horiz_px, horiz_candidates = find_building_dimensions(img, "horizontal")
        vert_px, vert_candidates = find_building_dimensions(img, "vertical")
        
        if horiz_px == 0 or vert_px == 0:
            return jsonify({
                "error": "Failed to detect scale lines",
                "debug": {
                    "horizontal_candidates": len(horiz_candidates),
                    "vertical_candidates": len(vert_candidates)
                }
            }), 400
        
        px_per_mm_h = horiz_px / horiz_mm
        px_per_mm_v = vert_px / vert_mm
        px_per_mm = (px_per_mm_h + px_per_mm_v) / 2
        px_per_meter = px_per_mm * 1000
        mm_per_px = 1 / px_per_mm if px_per_mm > 0 else 0
        
        return jsonify({
            "status": "ok",
            "scale": {
                "px_per_mm": round(px_per_mm, 4),
                "px_per_meter": round(px_per_meter, 2),
                "mm_per_px": round(mm_per_px, 4),
                "scale_ratio": f"1:{int(mm_per_px)}"
            },
            "detected_dimensions": {
                "horizontal_px": horiz_px,
                "vertical_px": vert_px,
                "horizontal_mm_calculated": round(horiz_px / px_per_mm, 1),
                "vertical_mm_calculated": round(vert_px / px_per_mm, 1)
            }
        })
    
    except Exception as e:
        return jsonify({
            "error": "Processing failed",
            "message": str(e)
        }), 500

@app.route("/analyze_rooms", methods=["POST"])
def analyze_rooms():
    """
    IMPROVED: More lenient room detection
    """
    try:
        data = request.get_json(force=True)
        
        if "image" not in data or "dimensions_summary" not in data:
            return jsonify({"error": "Missing image or dimensions_summary"}), 400
        
        img = b64_to_cv(data["image"])
        horiz_mm = data["dimensions_summary"].get("horizontal_mm")
        vert_mm = data["dimensions_summary"].get("vertical_mm")
        
        if not horiz_mm or not vert_mm:
            return jsonify({"error": "Invalid dimensions_summary"}), 400
        
        # Calculate scale
        horiz_px, _ = find_building_dimensions(img, "horizontal")
        vert_px, _ = find_building_dimensions(img, "vertical")
        
        if horiz_px == 0 or vert_px == 0:
            return jsonify({"error": "Failed to detect scale"}), 400
        
        px_per_mm = ((horiz_px / horiz_mm) + (vert_px / vert_mm)) / 2
        
        # IMPROVED: Much lower minimum area threshold
        min_area = data.get("min_room_area_m2", 0.5)  # Default 0.5 m²
        min_area_px = min_area * 1_000_000 * (px_per_mm ** 2)
        
        print(f"DEBUG: min_area_m2={min_area}, min_area_px={min_area_px}, px_per_mm={px_per_mm}")
        
        rooms = detect_rooms(img, min_area_px=min_area_px)
        
        if not rooms:
            return jsonify({
                "status": "error",
                "error": "No rooms detected",
                "debug": {
                    "min_area_px": int(min_area_px),
                    "min_area_m2": min_area,
                    "px_per_mm": round(px_per_mm, 4),
                    "suggestion": "Try lowering min_room_area_m2 to 0.3 or check if image is valid floorplan"
                }
            }), 200  # Return 200 so n8n can read the debug info
        
        # Extract room names
        rooms = assign_room_names(img, rooms)
        
        # Calculate areas
        rooms = calculate_room_areas(rooms, px_per_mm)
        
        # Prepare response
        room_data = []
        for i, room in enumerate(rooms, 1):
            room_data.append({
                'room_number': i,
                'name': room['name'],
                'area_m2': room['area_m2'],
                'width_m': room['width_m'],
                'height_m': room['height_m'],
                'center': room['center']
            })
        
        return jsonify({
            "status": "ok",
            "total_rooms": len(rooms),
            "total_area_m2": round(sum(r['area_m2'] for r in rooms), 2),
            "scale": {
                "px_per_mm": round(px_per_mm, 4),
                "px_per_meter": round(px_per_mm * 1000, 2)
            },
            "rooms": room_data
        })
    
    except Exception as e:
        import traceback
        return jsonify({
            "error": "Analysis failed",
            "message": str(e),
            "traceback": traceback.format_exc()
        }), 500

@app.route("/analyze_rooms_excel", methods=["POST"])
def analyze_rooms_excel():
    """Excel export endpoint"""
    try:
        data = request.get_json(force=True)
        
        if "image" not in data or "dimensions_summary" not in data:
            return jsonify({"error": "Missing image or dimensions_summary"}), 400
        
        img = b64_to_cv(data["image"])
        horiz_mm = data["dimensions_summary"].get("horizontal_mm")
        vert_mm = data["dimensions_summary"].get("vertical_mm")
        
        if not horiz_mm or not vert_mm:
            return jsonify({"error": "Invalid dimensions_summary"}), 400
        
        horiz_px, _ = find_building_dimensions(img, "horizontal")
        vert_px, _ = find_building_dimensions(img, "vertical")
        
        if horiz_px == 0 or vert_px == 0:
            return jsonify({"error": "Failed to detect scale"}), 400
        
        px_per_mm = ((horiz_px / horiz_mm) + (vert_px / vert_mm)) / 2
        
        min_area = data.get("min_room_area_m2", 0.5)
        min_area_px = min_area * 1_000_000 * (px_per_mm ** 2)
        
        rooms = detect_rooms(img, min_area_px=min_area_px)
        
        if not rooms:
            return jsonify({"error": "No rooms detected"}), 400
        
        rooms = assign_room_names(img, rooms)
        rooms = calculate_room_areas(rooms, px_per_mm)
        
        scale_info = {
            'px_per_mm': round(px_per_mm, 4),
            'px_per_meter': round(px_per_mm * 1000, 2)
        }
        
        excel_file = create_excel_report(rooms, scale_info)
        filename = f"romanalyse_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        
        return send_file(
            excel_file,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name=filename
        )
    
    except Exception as e:
        return jsonify({
            "error": "Excel generation failed",
            "message": str(e)
        }), 500

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy", 
        "service": "opencv-floorplan-processor",
        "version": "2.0-improved",
        "endpoints": ["/process", "/analyze_rooms", "/analyze_rooms_excel"]
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False)
