import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque
import math

# ========================
# Configuration
# ========================
class Config:
    # Camera settings
    CAMERA_INDEX = 0
    FRAME_WIDTH = 1280
    FRAME_HEIGHT = 720
    
    # Hand detection settings
    MAX_HANDS = 1
    DETECTION_CONFIDENCE = 0.7
    TRACKING_CONFIDENCE = 0.7
    
    # Drawing settings
    MIN_BRUSH_SIZE = 3
    MAX_BRUSH_SIZE = 40
    ERASER_SIZE = 50
    PALM_ERASER_SIZE = 80
    SMOOTHING_WINDOW = 5
    
    # Shape recognition settings
    SHAPE_DETECTION_ENABLED = True
    STROKE_FREEZE_TIME = 1.5  # Seconds to wait before converting to shape
    MIN_POINTS_FOR_SHAPE = 15  # Reduced from 20 for better detection
    
    # Object moving settings
    SELECTION_THRESHOLD = 50  # Distance in pixels to select an object
    
    # UI settings - BEAUTIFUL MINIMAL DESIGN ✨
    PALETTE_HEIGHT = 0  # No top palette - floating instead
    TOOLBAR_WIDTH = 0  # No left toolbar
    BOTTOM_BAR_HEIGHT = 0  # No bottom bar
    ALPHA_CANVAS = 0.92  # More canvas visible
    ALPHA_FRAME = 0.08  # Less camera overlay
    
    # Modern glassmorphism colors 🎨
    BG_COLOR = (15, 15, 20)  # Deep dark
    GLASS_BG = (25, 25, 35, 180)  # Semi-transparent
    ACCENT_COLOR = (140, 110, 255)  # Purple neon
    SUCCESS_COLOR = (100, 255, 150)  # Green
    WARNING_COLOR = (255, 200, 100)  # Orange
    ERROR_COLOR = (255, 100, 100)  # Red
    TEXT_COLOR = (255, 255, 255)
    
    # Floating palette settings
    FLOATING_PALETTE_SIZE = 45
    FLOATING_PALETTE_SPACING = 10

# ========================
# Shape Recognition Functions
# ========================

def detect_shape(points):
    """Detect if drawn stroke is a recognizable shape - ENHANCED VERSION"""
    if len(points) < Config.MIN_POINTS_FOR_SHAPE:
        return None, points
    
    # Convert points to numpy array
    pts = np.array([(p[0], p[1]) for p in points], dtype=np.float32)
    
    # Get color and thickness from first point
    color = points[0][2]
    thickness = points[0][3]
    
    # Calculate perimeter and area
    contour = pts.reshape((-1, 1, 2)).astype(np.int32)
    perimeter = cv2.arcLength(contour, True)
    perimeter_open = cv2.arcLength(contour, False)
    
    # Calculate area (for closed shapes)
    area = abs(cv2.contourArea(contour))
    
    if perimeter < 80:  # CHANGED: Reduced from 100 to 80 - allow smaller shapes
        return None, points
    
    # Get bounding box
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = w / float(h) if h > 0 else 0
    
    # Check if endpoints are close (closed shape) - MADE MORE LENIENT
    start_point = pts[0]
    end_point = pts[-1]
    endpoint_distance = np.sqrt((start_point[0] - end_point[0])**2 + (start_point[1] - end_point[1])**2)
    
    # CHANGED: More lenient - allow bigger gaps (was 0.12, now 0.25)
    is_closed = endpoint_distance < perimeter * 0.25  # Allow up to 25% gap
    
    print(f"Debug - Gap distance: {endpoint_distance:.1f}, Perimeter: {perimeter:.1f}, Closed: {is_closed}")
    
    # Multiple approximation levels for better detection
    epsilon_very_tight = 0.01 * perimeter
    epsilon_tight = 0.02 * perimeter
    epsilon_medium = 0.035 * perimeter
    epsilon_loose = 0.05 * perimeter
    epsilon_very_loose = 0.08 * perimeter
    
    approx_very_tight = cv2.approxPolyDP(contour, epsilon_very_tight, True)
    approx_tight = cv2.approxPolyDP(contour, epsilon_tight, True)
    approx_medium = cv2.approxPolyDP(contour, epsilon_medium, True)
    approx_loose = cv2.approxPolyDP(contour, epsilon_loose, True)
    approx_very_loose = cv2.approxPolyDP(contour, epsilon_very_loose, True)
    
    num_vertices = {
        'very_tight': len(approx_very_tight),
        'tight': len(approx_tight),
        'medium': len(approx_medium),
        'loose': len(approx_loose),
        'very_loose': len(approx_very_loose)
    }
    
    print(f"Debug - Closed: {is_closed}, Area: {area:.0f}, Perimeter: {perimeter:.0f}")
    print(f"Vertices - VT:{num_vertices['very_tight']}, T:{num_vertices['tight']}, M:{num_vertices['medium']}, L:{num_vertices['loose']}, VL:{num_vertices['very_loose']}")
    print(f"Aspect Ratio: {aspect_ratio:.2f}, Endpoint dist: {endpoint_distance:.1f}")
    
    # ============================================
    # DETECT LINE (straight line) - Check first before closed shapes
    # ============================================
    if num_vertices['very_loose'] <= 2 or num_vertices['loose'] <= 2:
        start = pts[0]
        end = pts[-1]
        line_vec = end - start
        line_length = np.linalg.norm(line_vec)
        
        if line_length > 50:  # Minimum line length
            # Calculate maximum deviation from straight line
            max_deviation = 0
            for point in pts:
                vec_to_point = point - start
                if line_length > 0:
                    cross = abs(np.cross(line_vec, vec_to_point))
                    deviation = cross / line_length
                    max_deviation = max(max_deviation, deviation)
            
            straightness_ratio = max_deviation / line_length
            print(f"Line check - Length: {line_length:.1f}, Max deviation: {max_deviation:.1f}, Ratio: {straightness_ratio:.3f}")
            
            # CHANGED: More lenient line detection (was 0.08, now 0.12)
            if straightness_ratio < 0.12 or max_deviation < 30:  # Allow slightly curved lines
                print("✨ Detected: LINE")
                return "line", [(int(start[0]), int(start[1]), color, thickness),
                               (int(end[0]), int(end[1]), color, thickness)]
    
    # For closed or nearly closed shapes
    # CHANGED: More lenient - allow bigger gaps (was 50px, now 100px)
    if is_closed or endpoint_distance < 100:
        
        # ============================================
        # DETECT CIRCLE
        # ============================================
        if area > 300:  # CHANGED: Reduced from 500 to 300 - allow smaller circles
            (cx, cy), radius = cv2.minEnclosingCircle(contour)
            circle_area = np.pi * radius * radius
            circularity = area / (circle_area + 1e-6) if circle_area > 0 else 0
            
            # Compactness test
            compactness = (4 * np.pi * area) / (perimeter * perimeter + 1e-6)
            
            print(f"Circle check - Circularity: {circularity:.3f}, Compactness: {compactness:.3f}, Aspect: {aspect_ratio:.2f}")
            
            # More lenient circle detection
            # CHANGED: More lenient circle detection
            if (0.5 < circularity < 1.6 and 
                0.4 < compactness < 1.5 and 
                0.6 < aspect_ratio < 1.6):
                print("✨ Detected: CIRCLE")
                return "circle", [("circle", int(cx), int(cy), int(radius), color, thickness)]
        
        # ============================================
        # DETECT RECTANGLE/SQUARE (Check multiple approximations)
        # ============================================
        for approx_name, num_v in [('medium', num_vertices['medium']), 
                                     ('loose', num_vertices['loose']),
                                     ('tight', num_vertices['tight'])]:
            if num_v == 4:
                if approx_name == 'medium':
                    corners = approx_medium.reshape(-1, 2)
                elif approx_name == 'loose':
                    corners = approx_loose.reshape(-1, 2)
                else:
                    corners = approx_tight.reshape(-1, 2)
                
                # Calculate angles at corners
                angles = []
                for i in range(4):
                    p1 = corners[i]
                    p2 = corners[(i+1)%4]
                    p3 = corners[(i+2)%4]
                    
                    v1 = p1 - p2
                    v2 = p3 - p2
                    
                    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
                    cos_angle = np.clip(cos_angle, -1.0, 1.0)
                    angle = np.arccos(cos_angle)
                    angles.append(np.degrees(angle))
                
                print(f"Rectangle check ({approx_name}) - Angles: {[f'{a:.1f}' for a in angles]}")
                
                # CHANGED: Even more lenient angle checking
                angles_close_to_90 = sum(1 for a in angles if 50 < a < 130)
                
                # CHANGED: Accept even with 2 good angles
                if angles_close_to_90 >= 2:
                    # CHANGED: More lenient aspect ratio
                    if 0.7 < aspect_ratio < 1.4:
                        print("✨ Detected: SQUARE")
                        detected_shape = "square"
                    else:
                        print("✨ Detected: RECTANGLE")
                        detected_shape = "rectangle"
                    
                    shape_points = [(int(pt[0]), int(pt[1]), color, thickness) for pt in corners]
                    shape_points.append(shape_points[0])
                    return detected_shape, shape_points
        
        # ============================================
        # DETECT TRIANGLE
        # ============================================
        for approx_name, num_v in [('medium', num_vertices['medium']), 
                                     ('loose', num_vertices['loose']),
                                     ('tight', num_vertices['tight']),
                                     ('very_loose', num_vertices['very_loose'])]:
            if num_v == 3:
                if approx_name == 'medium':
                    corners = approx_medium.reshape(-1, 2)
                elif approx_name == 'loose':
                    corners = approx_loose.reshape(-1, 2)
                elif approx_name == 'tight':
                    corners = approx_tight.reshape(-1, 2)
                else:
                    corners = approx_very_loose.reshape(-1, 2)
                
                # Calculate angles
                angles = []
                for i in range(3):
                    p1 = corners[i]
                    p2 = corners[(i+1)%3]
                    p3 = corners[(i+2)%3]
                    
                    v1 = p1 - p2
                    v2 = p3 - p2
                    
                    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
                    cos_angle = np.clip(cos_angle, -1.0, 1.0)
                    angle = np.arccos(cos_angle)
                    angles.append(np.degrees(angle))
                
                angle_sum = sum(angles)
                print(f"Triangle check ({approx_name}) - Angles: {[f'{a:.1f}' for a in angles]}, Sum: {angle_sum:.1f}")
                
                if 150 < angle_sum < 210:  # More lenient
                    print("✨ Detected: TRIANGLE")
                    shape_points = [(int(pt[0]), int(pt[1]), color, thickness) for pt in corners]
                    shape_points.append(shape_points[0])
                    return "triangle", shape_points
        
        # ============================================
        # DETECT ELLIPSE (after ruling out circle)
        # ============================================
        if len(contour) >= 5 and area > 800:
            try:
                ellipse = cv2.fitEllipse(contour)
                (cx, cy), (ma, mi), angle = ellipse
                
                # Ensure ma > mi
                if mi > ma:
                    ma, mi = mi, ma
                
                axis_ratio = ma / (mi + 1e-6)
                
                # Compactness for ellipse
                compactness = (4 * np.pi * area) / (perimeter * perimeter + 1e-6)
                
                print(f"Ellipse check - Axis ratio: {axis_ratio:.2f}, Compactness: {compactness:.3f}")
                
                if (0.5 < compactness < 1.2 and 
                    1.3 < axis_ratio < 4.0 and
                    ma > 30 and mi > 20):
                    print("✨ Detected: ELLIPSE")
                    return "ellipse", [("ellipse", (int(cx), int(cy)), (int(ma/2), int(mi/2)), int(angle), color, thickness)]
            except:
                pass
        
        # ============================================
        # DETECT PENTAGON
        # ============================================
        for approx_name, num_v in [('medium', num_vertices['medium']), 
                                     ('loose', num_vertices['loose']),
                                     ('tight', num_vertices['tight'])]:
            if num_v == 5:
                if approx_name == 'medium':
                    corners = approx_medium.reshape(-1, 2)
                elif approx_name == 'loose':
                    corners = approx_loose.reshape(-1, 2)
                else:
                    corners = approx_tight.reshape(-1, 2)
                
                print("✨ Detected: PENTAGON")
                shape_points = [(int(pt[0]), int(pt[1]), color, thickness) for pt in corners]
                shape_points.append(shape_points[0])
                return "pentagon", shape_points
        
        # ============================================
        # DETECT HEXAGON
        # ============================================
        for approx_name, num_v in [('medium', num_vertices['medium']), 
                                     ('loose', num_vertices['loose']),
                                     ('tight', num_vertices['tight'])]:
            if num_v == 6:
                if approx_name == 'medium':
                    corners = approx_medium.reshape(-1, 2)
                elif approx_name == 'loose':
                    corners = approx_loose.reshape(-1, 2)
                else:
                    corners = approx_tight.reshape(-1, 2)
                
                print("✨ Detected: HEXAGON")
                shape_points = [(int(pt[0]), int(pt[1]), color, thickness) for pt in corners]
                shape_points.append(shape_points[0])
                return "hexagon", shape_points
        
        # ============================================
        # DETECT GENERAL POLYGON
        # ============================================
        for approx_name, num_v in [('medium', num_vertices['medium']), 
                                     ('loose', num_vertices['loose'])]:
            if 7 <= num_v <= 12:
                if approx_name == 'medium':
                    corners = approx_medium.reshape(-1, 2)
                else:
                    corners = approx_loose.reshape(-1, 2)
                
                print(f"✨ Detected: POLYGON ({num_v} sides)")
                shape_points = [(int(pt[0]), int(pt[1]), color, thickness) for pt in corners]
                shape_points.append(shape_points[0])
                return f"polygon_{num_v}", shape_points
    
    # No shape detected
    print("❌ No shape detected - keeping original stroke")
    return None, points


def draw_shape_on_canvas(canvas, shape_type, shape_points):
    """Draw the recognized shape on canvas"""
    if not shape_points:
        return
    
    if shape_type == "circle":
        _, cx, cy, radius, color, thickness = shape_points[0]
        cv2.circle(canvas, (cx, cy), radius, color, thickness)
    
    elif shape_type == "ellipse":
        _, center, axes, angle, color, thickness = shape_points[0]
        cv2.ellipse(canvas, center, axes, angle, 0, 360, color, thickness)
    
    elif shape_type in ["line", "rectangle", "square", "triangle", "pentagon", "hexagon"] or "polygon" in shape_type:
        # Draw lines between points
        for i in range(1, len(shape_points)):
            x1, y1, col, thick = shape_points[i-1]
            x2, y2, _, _ = shape_points[i]
            cv2.line(canvas, (x1, y1), (x2, y2), col, thick)
    
    else:
        # Draw original stroke
        for i in range(1, len(shape_points)):
            x1, y1, col, thick = shape_points[i-1]
            x2, y2, _, _ = shape_points[i]
            cv2.line(canvas, (x1, y1), (x2, y2), col, thick)


# ========================
# Object Moving Functions  
# ========================

def get_shape_center(shape_points):
    """Get center point of a shape for selection"""
    if not shape_points or len(shape_points) == 0:
        return None
    
    if shape_points[0][0] == "circle":
        _, cx, cy, radius, _, _ = shape_points[0]
        return (cx, cy)
    
    elif shape_points[0][0] == "ellipse":
        _, center, axes, angle, _, _ = shape_points[0]
        return center
    
    else:
        # For lines and polygons - get centroid
        points = [(p[0], p[1]) for p in shape_points if isinstance(p[0], int)]
        if not points:
            return None
        
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        return (int(sum(xs) / len(xs)), int(sum(ys) / len(ys)))


def is_point_near_shape(x, y, shape_points, threshold=50):
    """Check if a point is near a shape's center"""
    center = get_shape_center(shape_points)
    if not center:
        return False
    
    cx, cy = center
    distance = np.sqrt((x - cx)**2 + (y - cy)**2)
    return distance < threshold


def move_shape(shape_points, dx, dy):
    """Move a shape by dx, dy offset"""
    if not shape_points or len(shape_points) == 0:
        return shape_points
    
    if shape_points[0][0] == "circle":
        _, cx, cy, radius, color, thickness = shape_points[0]
        return [("circle", cx + dx, cy + dy, radius, color, thickness)]
    
    elif shape_points[0][0] == "ellipse":
        _, center, axes, angle, color, thickness = shape_points[0]
        cx, cy = center
        return [("ellipse", (cx + dx, cy + dy), axes, angle, color, thickness)]
    
    else:
        # For lines and polygons
        moved_points = []
        for point in shape_points:
            if isinstance(point[0], int):  # Regular point
                x, y, col, thick = point
                moved_points.append((x + dx, y + dy, col, thick))
            else:
                moved_points.append(point)
        return moved_points


# ========================
# UI Helper Functions
# ========================
def create_gradient_bg(h, w):
    """Create modern gradient background"""
    gradient = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(h):
        ratio = i / h
        color = tuple(int(c * (1 - ratio * 0.3)) for c in [45, 45, 50])
        gradient[i, :] = color
    return gradient

def draw_rounded_rect(img, pt1, pt2, color, thickness=-1, radius=15):
    """Draw rounded rectangle"""
    x1, y1 = pt1
    x2, y2 = pt2
    
    cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
    cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, thickness)
    
    cv2.circle(img, (x1 + radius, y1 + radius), radius, color, thickness)
    cv2.circle(img, (x2 - radius, y1 + radius), radius, color, thickness)
    cv2.circle(img, (x1 + radius, y2 - radius), radius, color, thickness)
    cv2.circle(img, (x2 - radius, y2 - radius), radius, color, thickness)

def draw_icon_undo(img, x, y, size, color):
    center_x, center_y = x + size // 2, y + size // 2
    radius = size // 3
    cv2.ellipse(img, (center_x, center_y), (radius, radius), 0, 90, 270, color, 2)
    pts = np.array([[center_x - radius, center_y], 
                    [center_x - radius + 8, center_y - 8],
                    [center_x - radius + 8, center_y + 8]], np.int32)
    cv2.fillPoly(img, [pts], color)

def draw_icon_redo(img, x, y, size, color):
    center_x, center_y = x + size // 2, y + size // 2
    radius = size // 3
    cv2.ellipse(img, (center_x, center_y), (radius, radius), 0, 270, 90, color, 2)
    pts = np.array([[center_x + radius, center_y], 
                    [center_x + radius - 8, center_y - 8],
                    [center_x + radius - 8, center_y + 8]], np.int32)
    cv2.fillPoly(img, [pts], color)

def draw_icon_clear(img, x, y, size, color):
    center_x, center_y = x + size // 2, y + size // 2
    cv2.rectangle(img, (center_x - 8, center_y - 5), (center_x + 8, center_y + 10), color, 2)
    cv2.line(img, (center_x - 10, center_y - 5), (center_x + 10, center_y - 5), color, 2)
    cv2.line(img, (center_x - 6, center_y - 8), (center_x + 6, center_y - 8), color, 2)
    cv2.line(img, (center_x - 4, center_y), (center_x - 4, center_y + 6), color, 1)
    cv2.line(img, (center_x, center_y), (center_x, center_y + 6), color, 1)
    cv2.line(img, (center_x + 4, center_y), (center_x + 4, center_y + 6), color, 1)

def draw_icon_save(img, x, y, size, color):
    center_x, center_y = x + size // 2, y + size // 2
    cv2.rectangle(img, (center_x - 8, center_y - 8), (center_x + 8, center_y + 10), color, 2)
    cv2.rectangle(img, (center_x - 8, center_y - 8), (center_x + 8, center_y - 2), color, -1)
    cv2.rectangle(img, (center_x - 4, center_y - 8), (center_x + 4, center_y - 2), (25, 25, 30), -1)

def draw_icon_shapes(img, x, y, size, color):
    center_x, center_y = x + size // 2, y + size // 2
    cv2.rectangle(img, (center_x - 10, center_y - 8), (center_x - 2, center_y), color, 1)
    cv2.circle(img, (center_x + 6, center_y - 4), 5, color, 1)
    pts = np.array([[center_x - 6, center_y + 10], 
                    [center_x + 2, center_y + 10],
                    [center_x - 2, center_y + 3]], np.int32)
    cv2.polylines(img, [pts], True, color, 1)

def draw_button(img, x, y, w, h, text, icon_type=None, enabled=True, selected=False):
    if not enabled:
        bg_color = (35, 35, 40)
        text_color = (80, 80, 90)
    elif selected:
        bg_color = Config.HIGHLIGHT_COLOR
        text_color = (255, 255, 255)
    else:
        bg_color = (50, 50, 60)
        text_color = (220, 220, 230)
    
    draw_rounded_rect(img, (x, y), (x + w, y + h), bg_color, -1, radius=8)
    
    if enabled:
        border_color = (70, 70, 80) if not selected else (255, 255, 255)
        draw_rounded_rect(img, (x, y), (x + w, y + h), border_color, 2, radius=8)
    
    icon_size = 30
    icon_x = x + 12
    icon_y = y + (h - icon_size) // 2
    
    icon_color = text_color if enabled else (60, 60, 70)
    
    if icon_type == "undo":
        draw_icon_undo(img, icon_x, icon_y, icon_size, icon_color)
    elif icon_type == "redo":
        draw_icon_redo(img, icon_x, icon_y, icon_size, icon_color)
    elif icon_type == "clear":
        draw_icon_clear(img, icon_x, icon_y, icon_size, icon_color)
    elif icon_type == "save":
        draw_icon_save(img, icon_x, icon_y, icon_size, icon_color)
    elif icon_type == "shapes":
        draw_icon_shapes(img, icon_x, icon_y, icon_size, icon_color)
    
    text_x = x + 50
    text_y = y + h // 2 + 6
    cv2.putText(img, text, (text_x, text_y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.55, text_color, 2, cv2.LINE_AA)

def draw_color_palette(img, x, y, w, h, colors, color_names, current_idx):
    """Beautiful floating color palette at bottom center with glassmorphism ✨"""
    img_h, img_w = img.shape[:2]
    
    # Palette dimensions
    color_size = Config.FLOATING_PALETTE_SIZE
    spacing = Config.FLOATING_PALETTE_SPACING
    num_colors = len(colors)
    palette_width = (color_size + spacing) * num_colors + 20
    palette_height = 70
    
    # Position at bottom center
    palette_x = (img_w - palette_width) // 2
    palette_y = img_h - palette_height - 15
    
    # Glassmorphism background
    overlay = img.copy()
    
    # Rounded rectangle background
    cv2.rectangle(overlay, 
                 (palette_x - 10, palette_y), 
                 (palette_x + palette_width + 10, palette_y + palette_height),
                 (20, 20, 30), -1)
    
    # Blend with transparency for glass effect
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
    
    # Neon border glow
    cv2.rectangle(img, 
                 (palette_x - 10, palette_y), 
                 (palette_x + palette_width + 10, palette_y + palette_height),
                 Config.ACCENT_COLOR, 2)
    
    # Draw color circles
    for i, (col, name) in enumerate(zip(colors, color_names)):
        cx = palette_x + 10 + i * (color_size + spacing) + color_size // 2
        cy = palette_y + palette_height // 2
        center = (cx, cy)
        
        # Animated glow for selected color
        if i == current_idx:
            for r in range(color_size // 2 + 12, color_size // 2, -2):
                alpha = (color_size // 2 + 12 - r) / 12
                glow_color = tuple(int(c * 0.7 + 200 * 0.3) for c in col)
                cv2.circle(img, center, r, glow_color, 1)
        
        # Main color circle
        cv2.circle(img, center, color_size // 2, col, -1)
        
        # White ring
        cv2.circle(img, center, color_size // 2 + 1, (255, 255, 255), 2)
        
        # Selection indicator - neon ring
        if i == current_idx:
            cv2.circle(img, center, color_size // 2 + 6, Config.ACCENT_COLOR, 3)
            # Small dot indicator below
            cv2.circle(img, (cx, cy + color_size // 2 + 12), 3, Config.ACCENT_COLOR, -1)

def draw_toolbar(img, x, y, w, h, brush_size, strokes_count, redo_count, shape_mode, freeze_time):
    """Minimal floating brush size indicator (top left) ✨"""
    img_h, img_w = img.shape[:2]
    
    # Floating glass bubble - top left for brush size
    indicator_x = 20
    indicator_y = 20
    indicator_w = 100
    indicator_h = 45
    
    # Glassmorphism background
    overlay = img.copy()
    cv2.rectangle(overlay, 
                 (indicator_x, indicator_y), 
                 (indicator_x + indicator_w, indicator_y + indicator_h),
                 (20, 20, 30), -1)
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
    
    # Border
    cv2.rectangle(img, 
                 (indicator_x, indicator_y), 
                 (indicator_x + indicator_w, indicator_y + indicator_h),
                 Config.ACCENT_COLOR, 1)
    
    # Brush size text
    size_text = f"{brush_size}px"
    cv2.putText(img, size_text, (indicator_x + 10, indicator_y + 28),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Freeze timer display (analyzing shape) - centered
    if freeze_time > 0:
        timer_x = img_w // 2 - 80
        timer_y = 30
        
        progress = freeze_time / Config.STROKE_FREEZE_TIME
        
        # Glass background
        overlay = img.copy()
        cv2.rectangle(overlay, (timer_x, timer_y - 10), 
                     (timer_x + 160, timer_y + 30),
                     (20, 20, 30), -1)
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
        
        # Analyzing text
        cv2.putText(img, "ANALYZING...", (timer_x + 10, timer_y + 8),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, Config.WARNING_COLOR, 1, cv2.LINE_AA)
        
        # Progress bar
        bar_y = timer_y + 15
        bar_w = 140
        cv2.rectangle(img, (timer_x + 10, bar_y), 
                     (timer_x + 10 + bar_w, bar_y + 6), 
                     (40, 40, 50), -1)
        
        fill_w = int(bar_w * progress)
        cv2.rectangle(img, (timer_x + 10, bar_y), 
                     (timer_x + 10 + fill_w, bar_y + 6), 
                     Config.ACCENT_COLOR, -1)

def draw_status_bar(img, x, y, w, h, mode, show_landmarks, erasing_mode, shape_mode, move_mode=False):
    """Minimal floating status indicators - top left & top right ✨"""
    img_h, img_w = img.shape[:2]
    
    # Mode colors
    mode_colors = {
        "DRAW": Config.SUCCESS_COLOR,
        "SELECT": Config.WARNING_COLOR,
        "ERASE": Config.ERROR_COLOR,
        "MOVE": (200, 150, 255),
        "NONE": (150, 150, 150)
    }
    
    # Determine display mode
    if erasing_mode:
        display_mode = "ERASE"
    elif move_mode:
        display_mode = "MOVE"
    else:
        display_mode = mode
        
    mode_color = mode_colors.get(display_mode, (150, 150, 150))
    
    # Animated mode indicator - top left (next to brush size)
    dot_x = 140
    dot_y = 42
    
    # Pulsing outer ring
    cv2.circle(img, (dot_x, dot_y), 16, mode_color, 2)
    # Solid center
    cv2.circle(img, (dot_x, dot_y), 10, mode_color, -1)
    
    # Mode badges - top right corner
    badge_x = img_w - 100
    badge_y = 20
    
    # SHAPES badge (when enabled)
    if shape_mode:
        overlay = img.copy()
        cv2.rectangle(overlay, (badge_x - 10, badge_y), 
                     (badge_x + 85, badge_y + 35),
                     (20, 20, 30), -1)
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
        
        cv2.rectangle(img, (badge_x - 10, badge_y), 
                     (badge_x + 85, badge_y + 35),
                     Config.ACCENT_COLOR, 2)
        
        cv2.putText(img, "SHAPES", (badge_x, badge_y + 23),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, Config.ACCENT_COLOR, 2, cv2.LINE_AA)
        
        badge_x -= 110
    
    # MOVE badge (when enabled)
    if move_mode:
        overlay = img.copy()
        cv2.rectangle(overlay, (badge_x - 10, badge_y), 
                     (badge_x + 75, badge_y + 35),
                     (20, 20, 30), -1)
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
        
        cv2.rectangle(img, (badge_x - 10, badge_y), 
                     (badge_x + 75, badge_y + 35),
                     (255, 150, 255), 2)
        
        cv2.putText(img, "MOVE", (badge_x, badge_y + 23),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 150, 255), 2, cv2.LINE_AA)

# ========================
# Webcam setup
# ========================
cap = cv2.VideoCapture(Config.CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.FRAME_HEIGHT)

# ========================
# MediaPipe setup
# ========================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    max_num_hands=Config.MAX_HANDS,
    min_detection_confidence=Config.DETECTION_CONFIDENCE,
    min_tracking_confidence=Config.TRACKING_CONFIDENCE
)

# ========================
# Canvas and strokes
# ========================
canvas = None
prev_points = deque(maxlen=Config.SMOOTHING_WINDOW)
strokes = []
current_stroke = []
redo_stack = []

# Time-based stroke freezing
stroke_end_time = 0
pending_stroke = None

# ========================
# Colors and tools
# ========================
colors = [
    (255, 0, 255),   # Purple
    (255, 0, 0),     # Blue
    (0, 255, 0),     # Green
    (0, 255, 255),   # Yellow
    (0, 165, 255),   # Orange
    (0, 127, 255),   # Red
    (255, 255, 255), # White
    (0, 0, 0)        # Eraser
]

color_names = ["PURPLE", "BLUE", "GREEN", "YELLOW", "ORANGE", "RED", "WHITE", "ERASER"]
current_color = colors[0]
current_color_idx = 0

# ========================
# Drawing modes
# ========================
show_hand_landmarks = True
brush_size_mode = "FIXED"
fixed_brush_size = 10
is_fullscreen = False
shape_recognition_enabled = True

# Object moving mode
move_mode_enabled = False
selected_object_idx = None
last_move_pos = None

# ========================
# Window setup
# ========================
window_name = "Air Canvas Pro - Shapes & Moving Objects"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 1280, 720)

# ========================
# Helper Functions
# ========================

def fingers_up(hand):
    landmarks = hand.landmark
    fingers = []
    
    if landmarks[4].x < landmarks[3].x:
        fingers.append(landmarks[4].x < landmarks[2].x)
    else:
        fingers.append(landmarks[4].x > landmarks[2].x)
    
    finger_tips = [8, 12, 16, 20]
    finger_pips = [6, 10, 14, 18]
    
    for tip, pip in zip(finger_tips, finger_pips):
        fingers.append(landmarks[tip].y < landmarks[pip].y)
    
    return fingers

def is_palm_open(hand):
    finger_status = fingers_up(hand)
    return all(finger_status)

def get_smoothed_point(x, y):
    prev_points.append((x, y))
    if len(prev_points) < 2:
        return x, y
    
    avg_x = int(np.mean([p[0] for p in prev_points]))
    avg_y = int(np.mean([p[1] for p in prev_points]))
    return avg_x, avg_y

def calculate_brush_size(hand, w, h):
    if brush_size_mode == "FIXED":
        return fixed_brush_size
    
    index_x = int(hand.landmark[8].x * w)
    index_y = int(hand.landmark[8].y * h)
    thumb_x = int(hand.landmark[4].x * w)
    thumb_y = int(hand.landmark[4].y * h)
    
    distance = np.hypot(index_x - thumb_x, index_y - thumb_y)
    thickness = int(np.interp(distance, [20, 200], [Config.MIN_BRUSH_SIZE, Config.MAX_BRUSH_SIZE]))
    
    return thickness

def get_palm_center(hand, w, h):
    wrist_x = int(hand.landmark[0].x * w)
    wrist_y = int(hand.landmark[0].y * h)
    middle_base_x = int(hand.landmark[9].x * w)
    middle_base_y = int(hand.landmark[9].y * h)
    
    center_x = (wrist_x + middle_base_x) // 2
    center_y = (wrist_y + middle_base_y) // 2
    
    return center_x, center_y

def redraw_canvas(h, w):
    new_canvas = np.zeros((h, w, 3), dtype=np.uint8)
    
    for stroke in strokes:
        if len(stroke) < 1:
            continue
        
        if stroke[0][0] == "circle":
            _, cx, cy, radius, color, thickness = stroke[0]
            cv2.circle(new_canvas, (cx, cy), radius, color, thickness)
        elif stroke[0][0] == "ellipse":
            _, center, axes, angle, color, thickness = stroke[0]
            cv2.ellipse(new_canvas, center, axes, angle, 0, 360, color, thickness)
        else:
            for i in range(1, len(stroke)):
                x1, y1, col, thick = stroke[i-1]
                x2, y2, _, _ = stroke[i]
                cv2.line(new_canvas, (x1, y1), (x2, y2), col, thick)
    
    return new_canvas

def save_canvas(canvas):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"AirCanvas_{timestamp}.png"
    
    white_bg = np.ones_like(canvas) * 255
    mask = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    white_bg[mask > 0] = canvas[mask > 0]
    
    cv2.imwrite(filename, white_bg)
    print(f"✓ Canvas saved as {filename}")
    return filename

def process_gestures(hand, w, h):
    """Process hand gestures including palm erasing"""
    finger_status = fingers_up(hand)
    
    # Check for palm erase first
    if is_palm_open(hand):
        palm_x, palm_y = get_palm_center(hand, w, h)
        smooth_x, smooth_y = get_smoothed_point(palm_x, palm_y)
        return "ERASE", smooth_x, smooth_y, Config.PALM_ERASER_SIZE
    
    index_x = int(hand.landmark[8].x * w)
    index_y = int(hand.landmark[8].y * h)
    smooth_x, smooth_y = get_smoothed_point(index_x, index_y)
    
    thumb, index, middle, ring, pinky = finger_status
    
    # 3-finger gesture for move/select (index + middle + ring)
    if index and middle and ring and not pinky:
        return "MOVE_SELECT", smooth_x, smooth_y, None
    
    # Selection mode: Index + Middle finger
    elif index and middle and not ring and not pinky:
        return "SELECT", smooth_x, smooth_y, None
    
    # Draw mode: Only index finger
    elif index and not middle and not ring and not pinky:
        brush_size = calculate_brush_size(hand, w, h)
        if current_color == (0, 0, 0):
            brush_size = Config.ERASER_SIZE
        return "DRAW", smooth_x, smooth_y, brush_size
    
    else:
        return "NONE", smooth_x, smooth_y, None

# ========================
# Main Loop
# ========================
print("✨🎨 AIR CANVAS PRO - BEAUTIFUL EDITION 🎨✨")
print("=" * 70)
print("🌟 FEATURES:")
print("  ✨ Minimal Glassmorphism UI")
print("  🎨 Floating Color Palette")
print("  📐 Auto Shape Detection (Circles, Rectangles, Triangles, etc.)")
print("  🔄 Drag & Move Objects")
print("  💜 Purple Neon Theme")
print("\n📝 HOW TO USE:")
print("  DRAW MODE:")
print("    • Index finger to draw")
print("    • Remove hand, wait 1.5s → shape auto-detects!")
print("  COLOR SELECTION:")
print("    • Index + Middle fingers → point at floating palette (bottom)")
print("  MOVE MODE (Press M):")
print("    • 3 fingers (I+M+R) → select object (yellow highlight)")
print("    • Index only → drag selected object")
print("    • Remove hand → drop object")
print("\n⌨️  KEYBOARD SHORTCUTS:")
print("  G: Shapes ON/OFF  │  M: Move Mode  │  C: Clear")
print("  U: Undo │ R: Redo │  S: Save  │  +/-: Brush Size")
print("  H: Hand Landmarks │  F: Fullscreen │  Q: Quit")
print("=" * 70)
print("🚀 Launching beautiful Air Canvas...")
print("=" * 70)

while True:
    success, frame = cap.read()
    if not success:
        break
    
    current_time = time.time()
    
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    
    if canvas is None:
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
    
    ui_overlay = create_gradient_bg(h, w)
    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    
    mode = "NONE"
    current_brush_size = fixed_brush_size
    erasing_with_palm = False
    
    # Check if we should process pending stroke
    freeze_timer = 0
    if pending_stroke and current_time - stroke_end_time >= Config.STROKE_FREEZE_TIME:
        # Time to analyze the shape!
        if shape_recognition_enabled and len(pending_stroke) >= Config.MIN_POINTS_FOR_SHAPE:
            detected_shape, shape_points = detect_shape(pending_stroke)
            if detected_shape:
                canvas = redraw_canvas(h, w)
                draw_shape_on_canvas(canvas, detected_shape, shape_points)
                strokes.append(shape_points)
            else:
                print("No shape detected - keeping original")
                strokes.append(pending_stroke)
        else:
            strokes.append(pending_stroke)
        
        pending_stroke = None
        stroke_end_time = 0
    elif pending_stroke:
        freeze_timer = current_time - stroke_end_time
    
    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:
            
            if show_hand_landmarks:
                mp_draw.draw_landmarks(
                    frame, 
                    hand, 
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
            
            mode, x, y, brush_size = process_gestures(hand, w, h)
            
            if brush_size:
                current_brush_size = brush_size
            
            if mode == "ERASE":
                erasing_with_palm = True
                for r in range(Config.PALM_ERASER_SIZE, Config.PALM_ERASER_SIZE - 20, -4):
                    alpha = (Config.PALM_ERASER_SIZE - r) / 20
                    color = tuple(int(255 * alpha) for _ in range(3))
                    cv2.circle(frame, (x, y), r, color, 2)
                
                cv2.circle(frame, (x, y), Config.PALM_ERASER_SIZE, (255, 100, 100), 3)
                cv2.circle(frame, (x, y), Config.PALM_ERASER_SIZE // 2, (255, 150, 150), -1)
                
                cv2.circle(canvas, (x, y), Config.PALM_ERASER_SIZE, (0, 0, 0), -1)
                
                # Cancel pending stroke
                if pending_stroke:
                    strokes.append(pending_stroke)
                    pending_stroke = None
                    stroke_end_time = 0
                
                if current_stroke:
                    current_stroke = []
                    redo_stack = []
                
                # Deselect object when erasing
                if selected_object_idx is not None:
                    selected_object_idx = None
                    last_move_pos = None
            
            elif mode == "MOVE_SELECT" and move_mode_enabled:
                # 3-finger gesture to select object
                cv2.circle(frame, (x, y), 20, (255, 255, 0), 2)
                cv2.circle(frame, (x, y), 3, (255, 255, 0), -1)
                
                # Try to select an object
                for idx in range(len(strokes) - 1, -1, -1):  # Check from newest to oldest
                    if is_point_near_shape(x, y, strokes[idx], Config.SELECTION_THRESHOLD):
                        selected_object_idx = idx
                        last_move_pos = (x, y)
                        print(f"🎯 Selected object {idx}")
                        
                        # Draw selection highlight
                        center = get_shape_center(strokes[idx])
                        if center:
                            cv2.circle(frame, center, 70, (255, 255, 0), 4)
                            cv2.putText(frame, "SELECTED", (center[0] - 50, center[1] - 80),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                        break
                
                # Cancel drawing
                if pending_stroke:
                    strokes.append(pending_stroke)
                    pending_stroke = None
                    stroke_end_time = 0
                if current_stroke:
                    current_stroke = []
            
            elif mode == "DRAW" and move_mode_enabled and selected_object_idx is not None:
                # Drag the selected object with index finger
                if last_move_pos:
                    dx = x - last_move_pos[0]
                    dy = y - last_move_pos[1]
                    
                    # Move the object
                    strokes[selected_object_idx] = move_shape(strokes[selected_object_idx], dx, dy)
                    
                    # Redraw canvas
                    canvas = redraw_canvas(h, w)
                    
                    last_move_pos = (x, y)
                    
                    # Show dragging indicator
                    cv2.circle(frame, (x, y), 25, (255, 255, 0), -1)
                    cv2.circle(frame, (x, y), 28, (255, 255, 255), 3)
                    cv2.putText(frame, "DRAGGING", (x - 50, y - 40),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            elif mode == "SELECT":
                cv2.circle(frame, (x, y), 15, current_color, cv2.FILLED)
                cv2.circle(frame, (x, y), 18, (255, 255, 255), 2)
                
                # Cancel pending stroke and current stroke
                if pending_stroke:
                    strokes.append(pending_stroke)
                    pending_stroke = None
                    stroke_end_time = 0
                
                if current_stroke:
                    current_stroke = []
                    redo_stack = []
                
                # Check if selecting color from floating palette (bottom of screen)
                num_colors = len(colors)
                color_size = Config.FLOATING_PALETTE_SIZE
                spacing = Config.FLOATING_PALETTE_SPACING
                palette_width = (color_size + spacing) * num_colors + 20
                palette_height = 70
                
                # Palette at bottom center
                palette_x = (w - palette_width) // 2
                palette_y = h - palette_height - 15
                
                # Check if y is in palette area
                if palette_y <= y <= palette_y + palette_height:
                    for i in range(num_colors):
                        cx = palette_x + 10 + i * (color_size + spacing) + color_size // 2
                        cy = palette_y + palette_height // 2
                        center = (cx, cy)
                        dist = np.hypot(x - center[0], y - center[1])
                        if dist < color_size // 2:
                            current_color = colors[i]
                            current_color_idx = i
                            print(f"🎨 Selected color: {color_names[i]}")
                            break
            
            elif mode == "DRAW":
                cv2.circle(frame, (x, y), current_brush_size, current_color, cv2.FILLED)
                cv2.circle(frame, (x, y), current_brush_size + 2, (255, 255, 255), 2)
                
                # If there was a pending stroke, commit it first
                if pending_stroke:
                    strokes.append(pending_stroke)
                    pending_stroke = None
                    stroke_end_time = 0
                
                if len(current_stroke) > 0:
                    prev_x, prev_y, _, _ = current_stroke[-1]
                    cv2.line(canvas, (prev_x, prev_y), (x, y), current_color, current_brush_size)
                
                current_stroke.append((x, y, current_color, current_brush_size))
            
            else:
                # Hand visible but not drawing - start freeze timer
                if current_stroke and not pending_stroke:
                    print("Hand visible but not drawing - starting freeze timer")
                    pending_stroke = current_stroke.copy()
                    current_stroke = []
                    stroke_end_time = current_time
                    redo_stack = []
    else:
        # No hand detected - IMMEDIATELY start freeze timer if we have a current stroke
        if current_stroke and not pending_stroke:
            print("Hand removed - starting freeze timer IMMEDIATELY")
            pending_stroke = current_stroke.copy()
            current_stroke = []
            stroke_end_time = current_time
            redo_stack = []
        
        # Deselect object when hand is removed
        if move_mode_enabled and selected_object_idx is not None:
            print("✅ Object dropped")
            selected_object_idx = None
            last_move_pos = None
        
        # Continue counting down if pending stroke exists
        # (This allows the countdown to continue even when hand is removed)
        
        prev_points.clear()
    
    combined = cv2.addWeighted(frame, Config.ALPHA_FRAME, canvas, Config.ALPHA_CANVAS, 0)
    combined = cv2.addWeighted(combined, 0.9, ui_overlay, 0.1, 0)
    
    # Beautiful minimal UI - floating elements only! ✨
    draw_color_palette(combined, 0, 0, w, h, colors, color_names, current_color_idx)
    
    draw_toolbar(combined, 0, 0, 0, h, current_brush_size, len(strokes), len(redo_stack), 
                shape_recognition_enabled, freeze_timer)
    
    draw_status_bar(combined, 0, 0, w, 0, mode, show_hand_landmarks, erasing_with_palm, 
                   shape_recognition_enabled, move_mode_enabled)
    
    # Draw cursor indicator when drawing
    if mode == "DRAW" and not erasing_with_palm and not move_mode_enabled:
        # Neon cursor ring
        cv2.circle(combined, (x, y), current_brush_size + 2, Config.ACCENT_COLOR, 1)
        cv2.circle(combined, (x, y), current_brush_size, current_color, 2)
    
    cv2.imshow(window_name, combined)
    
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('c') or key == ord('C'):
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        strokes = []
        redo_stack = []
        pending_stroke = None
        stroke_end_time = 0
        print("🗑️  Canvas cleared")
    
    elif key == ord('u') or key == ord('U'):
        if strokes:
            redo_stack.append(strokes.pop())
            canvas = redraw_canvas(h, w)
            print(f"↶ Undo (strokes: {len(strokes)})")
    
    elif key == ord('r') or key == ord('R'):
        if redo_stack:
            strokes.append(redo_stack.pop())
            canvas = redraw_canvas(h, w)
            print(f"↷ Redo (strokes: {len(strokes)})")
    
    elif key == ord('s') or key == ord('S'):
        save_canvas(canvas)
    
    elif key == ord('h') or key == ord('H'):
        show_hand_landmarks = not show_hand_landmarks
        print(f"👋 Hand landmarks: {'ON' if show_hand_landmarks else 'OFF'}")
    
    elif key == ord('+') or key == ord('='):
        fixed_brush_size = min(fixed_brush_size + 2, Config.MAX_BRUSH_SIZE)
        brush_size_mode = "FIXED"
        print(f"🖌️  Brush size: {fixed_brush_size}px")
    
    elif key == ord('-') or key == ord('_'):
        fixed_brush_size = max(fixed_brush_size - 2, Config.MIN_BRUSH_SIZE)
        brush_size_mode = "FIXED"
        print(f"🖌️  Brush size: {fixed_brush_size}px")
    
    elif key == ord('g') or key == ord('G'):
        shape_recognition_enabled = not shape_recognition_enabled
        status = "ON" if shape_recognition_enabled else "OFF"
        print(f"✨ Shape recognition: {status}")
    
    elif key == ord('m') or key == ord('M'):
        move_mode_enabled = not move_mode_enabled
        selected_object_idx = None  # Deselect when toggling
        last_move_pos = None
        status = "ON" if move_mode_enabled else "OFF"
        print(f"🔄 Move mode: {status}")
        if move_mode_enabled:
            print("   → Use 3 fingers (index+middle+ring) to SELECT object")
            print("   → Use index finger only to DRAG selected object")
            print("   → Remove hand to DROP object")
    
    elif key == ord('f') or key == ord('F'):
        if is_fullscreen:
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
            is_fullscreen = False
            print("📐 Normal window mode")
        else:
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            is_fullscreen = True
            print("🖥️  Fullscreen mode")
    
    elif key == ord('q') or key == ord('Q'):
        print("\n👋 Goodbye!")
        break

cap.release()
cv2.destroyAllWindows()