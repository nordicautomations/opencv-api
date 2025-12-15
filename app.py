import io, base64, math
from flask import Flask, request, jsonify, send_file
import numpy as np
import cv2
from PIL import Image
from collections import defaultdict
import pytesseract
import pandas as pd
from datetime import datetime
import tempfile
import os

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
# Room Detection & Analysis
# -------------------------

def detect_rooms(img, min_area_px=5000, max_area_px=None):
    """
    Detect individual rooms using contour detection
    Returns list of room contours with bounding boxes
    """
    binary, _ = preprocess_floorplan(img)
    
    # Invert so rooms are white
    inverted = cv2.bitwise_not(binary)
    
    # Close small gaps in walls
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(inverted, cv2.MORPH_CLOSE, kernel, iterations=3)
    
    # Find contours
    contours, hierarchy = cv2.findContours(closed, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    
    if max_area_px is None:
        max_area_px = img.shape[0] * img.shape[1] * 0.8  # 80% of image
    
    rooms = []
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        
        # Filter by area and ensure it's not a hole (hierarchy check)
        if min_area_px < area < max_area_px:
            # Check if it's an outer contour (not a hole)
            if hierarchy[0][i][3] == -1 or hierarchy[0][i][2] != -1:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Additional shape validation (not too thin/elongated)
                aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 0
                if aspect_ratio < 10:  # Not a corridor or line
                    rooms.append({
                        'contour': contour,
                        'area_px': area,
                        'bbox': (x, y, w, h),
                        'center': (x + w//2, y + h//2)
                    })
    
    # Sort by area (largest first)
    rooms.sort(key=lambda x: x['area_px'], reverse=True)
    
    return rooms

def extract_text_from_region(img, bbox, padding=10):
    """
    Extract text from a specific region using Tesseract OCR
    """
    x, y, w, h = bbox
    
    # Add padding
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(img.shape[1] - x, w + 2*padding)
    h = min(img.shape[0] - y, h + 2*padding)
    
    # Extract region
    roi = img[y:y+h, x:x+w]
    
    # Preprocess for better OCR
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
    
    # Increase contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # Threshold
    _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # OCR with Norwegian language support
    custom_config = r'--oem 3 --psm 6 -l nor+eng'
    text = pytesseract.image_to_string(thresh, config=custom_config)
    
    # Clean up text
    text = text.strip()
    text = ' '.join(text.split())  # Remove extra whitespace
    
    return text if text else "Unknown"

def assign_room_names(img, rooms):
    """
    Extract room names/labels from detected room regions
    """
    for room in rooms:
        x, y, w, h = room['bbox']
        
        # Try to extract text from the room center
        text = extract_text_from_region(img, room['bbox'], padding=20)
        
        # Clean and validate room name
        if text and len(text) > 1:
            # Remove common noise patterns
            text = text.replace('|', '').replace('_', ' ')
            room['name'] = text[:50]  # Limit length
        else:
            room['name'] = f"Rom {rooms.index(room) + 1}"
    
    return rooms

def calculate_room_areas(rooms, px_per_mm):
    """
    Calculate room areas in square meters
    """
    for room in rooms:
        area_px = room['area_px']
        
        # Convert pixels to mm²
        area_mm2 = area_px / (px_per_mm ** 2)
        
        # Convert to m²
        area_m2 = area_mm2 / 1_000_000
        
        room['area_m2'] = round(area_m2, 2)
        
        # Also calculate dimensions
        _, _, w_px, h_px = room['bbox']
        room['width_m'] = round((w_px / px_per_mm) / 1000, 2)
        room['height_m'] = round((h_px / px_per_mm) / 1000, 2)
    
    return rooms

def create_excel_report(rooms, scale_info, metadata=None):
    """
    Create Excel file with room data
    """
    # Prepare data for DataFrame
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
    
    # Create Excel file with multiple sheets
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Main data sheet
        df.to_excel(writer, sheet_name='Romanalyse', index=False)
        
        # Summary sheet
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
        
        # Auto-adjust column widths
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
    Complete room analysis endpoint
    Returns room data as JSON
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
        
        # Detect rooms
        min_area = data.get("min_room_area_m2", 2.0)  # Default 2 m²
        min_area_px = min_area * 1_000_000 * (px_per_mm ** 2)
        
        rooms = detect_rooms(img, min_area_px=min_area_px)
        
        if not rooms:
            return jsonify({
                "error": "No rooms detected",
                "suggestion": "Try adjusting min_room_area_m2 parameter"
            }), 400
        
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
        return jsonify({
            "error": "Analysis failed",
            "message": str(e)
        }), 500

@app.route("/analyze_rooms_excel", methods=["POST"])
def analyze_rooms_excel():
    """
    Complete room analysis with Excel export
    Returns Excel file
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
        
        # Detect rooms
        min_area = data.get("min_room_area_m2", 2.0)
        min_area_px = min_area * 1_000_000 * (px_per_mm ** 2)
        
        rooms = detect_rooms(img, min_area_px=min_area_px)
        
        if not rooms:
            return jsonify({"error": "No rooms detected"}), 400
        
        rooms = assign_room_names(img, rooms)
        rooms = calculate_room_areas(rooms, px_per_mm)
        
        # Create Excel file
        scale_info = {
            'px_per_mm': round(px_per_mm, 4),
            'px_per_meter': round(px_per_mm * 1000, 2)
        }
        
        excel_file = create_excel_report(rooms, scale_info)
        
        # Generate filename with timestamp
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
        "endpoints": ["/process", "/analyze_rooms", "/analyze_rooms_excel"]
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False)
