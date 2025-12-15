import io, base64, math
from flask import Flask, request, jsonify
import numpy as np
import cv2
from PIL import Image
from collections import defaultdict

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
    """
    Advanced preprocessing for architectural drawings
    - Removes noise
    - Enhances contrast
    - Binarizes intelligently
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Denoise while preserving edges
    denoised = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
    
    # Adaptive thresholding works better than Canny for architectural drawings
    binary = cv2.adaptiveThreshold(
        denoised, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 
        blockSize=15, 
        C=8
    )
    
    # Morphological operations to connect broken lines and remove small noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    
    return binary, denoised

def detect_main_structure_lines(binary_img, orientation="horizontal"):
    """
    Enhanced line detection using multiple methods
    Returns the most reliable wall measurements
    """
    height, width = binary_img.shape
    min_length = (width if orientation == "horizontal" else height) * 0.3
    
    # Method 1: Probabilistic Hough Transform with optimized parameters
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
    angle_tolerance = 15  # degrees
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        # Calculate angle
        angle = math.degrees(math.atan2(abs(y2 - y1), abs(x2 - x1)))
        
        if orientation == "horizontal":
            # Should be nearly horizontal (0° or 180°)
            if angle < angle_tolerance or angle > (180 - angle_tolerance):
                valid_lines.append({
                    'length': length,
                    'coords': (x1, y1, x2, y2),
                    'angle': angle,
                    'midpoint_y': (y1 + y2) / 2
                })
        else:  # vertical
            # Should be nearly vertical (90°)
            if abs(angle - 90) < angle_tolerance:
                valid_lines.append({
                    'length': length,
                    'coords': (x1, y1, x2, y2),
                    'angle': angle,
                    'midpoint_x': (x1 + x2) / 2
                })
    
    return valid_lines

def cluster_lines(lines, orientation="horizontal", tolerance=50):
    """
    Group parallel lines that likely represent the same wall
    Returns the longest line from each cluster
    """
    if not lines:
        return []
    
    clusters = defaultdict(list)
    position_key = 'midpoint_y' if orientation == "horizontal" else 'midpoint_x'
    
    # Sort lines by their position
    sorted_lines = sorted(lines, key=lambda x: x[position_key])
    
    # Group lines that are close together
    cluster_id = 0
    for line in sorted_lines:
        if not clusters or abs(line[position_key] - 
                               clusters[cluster_id][-1][position_key]) > tolerance:
            cluster_id += 1
        clusters[cluster_id].append(line)
    
    # Get longest line from each cluster
    representative_lines = []
    for cluster in clusters.values():
        longest = max(cluster, key=lambda x: x['length'])
        representative_lines.append(longest)
    
    return representative_lines

def find_building_dimensions(img, orientation="horizontal"):
    """
    Main function to find building dimensions using enhanced detection
    """
    binary, denoised = preprocess_floorplan(img)
    
    # Detect lines
    raw_lines = detect_main_structure_lines(binary, orientation)
    
    if not raw_lines:
        # Fallback: try with more aggressive edge detection
        edges = cv2.Canny(denoised, 30, 100)
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        raw_lines = detect_main_structure_lines(edges, orientation)
    
    if not raw_lines:
        return 0, []
    
    # Cluster parallel lines
    clustered_lines = cluster_lines(raw_lines, orientation)
    
    # Sort by length and return top candidates
    clustered_lines.sort(key=lambda x: x['length'], reverse=True)
    
    # Return longest line (likely outer wall) and all candidates for validation
    longest_length = int(clustered_lines[0]['length'])
    
    return longest_length, clustered_lines[:5]  # Top 5 for debugging

def validate_dimension_match(detected_px, expected_mm, px_per_mm_estimate, tolerance=0.15):
    """
    Validate if detected pixel dimension matches expected millimeter dimension
    Returns confidence score (0-1)
    """
    if detected_px == 0 or expected_mm == 0:
        return 0.0
    
    calculated_mm = detected_px / px_per_mm_estimate
    ratio = min(calculated_mm, expected_mm) / max(calculated_mm, expected_mm)
    
    # If within tolerance, return high confidence
    if ratio >= (1 - tolerance):
        return ratio
    return 0.0

# -------------------------
# API Endpoint
# -------------------------

@app.route("/process", methods=["POST"])
def process():
    try:
        data = request.get_json(force=True)
        
        # Validate input
        if "image" not in data or "dimensions_summary" not in data:
            return jsonify({"error": "Missing image or dimensions_summary"}), 400
        
        img = b64_to_cv(data["image"])
        horiz_mm = data["dimensions_summary"].get("horizontal_mm")
        vert_mm = data["dimensions_summary"].get("vertical_mm")
        
        if not horiz_mm or not vert_mm:
            return jsonify({"error": "Invalid dimensions_summary"}), 400
        
        # Detect dimensions with enhanced algorithm
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
        
        # Calculate scale
        px_per_mm_h = horiz_px / horiz_mm
        px_per_mm_v = vert_px / vert_mm
        px_per_mm = (px_per_mm_h + px_per_mm_v) / 2
        
        # Validate consistency
        scale_diff = abs(px_per_mm_h - px_per_mm_v) / px_per_mm
        
        # Calculate confidence scores
        horiz_confidence = validate_dimension_match(horiz_px, horiz_mm, px_per_mm, tolerance=0.15)
        vert_confidence = validate_dimension_match(vert_px, vert_mm, px_per_mm, tolerance=0.15)
        overall_confidence = (horiz_confidence + vert_confidence) / 2
        
        px_per_meter = px_per_mm * 1000
        mm_per_px = 1 / px_per_mm if px_per_mm > 0 else 0
        
        response = {
            "status": "ok",
            "scale": {
                "px_per_mm": round(px_per_mm, 4),
                "px_per_meter": round(px_per_meter, 2),
                "mm_per_px": round(mm_per_px, 4),
                "scale_ratio": f"1:{int(mm_per_px)}"
            },
            "confidence": {
                "overall": round(overall_confidence, 2),
                "horizontal": round(horiz_confidence, 2),
                "vertical": round(vert_confidence, 2),
                "scale_consistency": round(1 - scale_diff, 2)
            },
            "detected_dimensions": {
                "horizontal_px": horiz_px,
                "vertical_px": vert_px,
                "horizontal_mm_calculated": round(horiz_px / px_per_mm, 1),
                "vertical_mm_calculated": round(vert_px / px_per_mm, 1)
            },
            "debug": {
                "image_size": {"width": img.shape[1], "height": img.shape[0]},
                "horizontal_candidates_count": len(horiz_candidates),
                "vertical_candidates_count": len(vert_candidates),
                "top_horizontal_lengths": [int(c['length']) for c in horiz_candidates[:3]],
                "top_vertical_lengths": [int(c['length']) for c in vert_candidates[:3]]
            }
        }
        
        # Add warning if confidence is low
        if overall_confidence < 0.7:
            response["warning"] = "Low confidence in scale detection. Results may be inaccurate."
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({
            "error": "Processing failed",
            "message": str(e)
        }), 500

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "service": "opencv-floorplan-processor"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False)
