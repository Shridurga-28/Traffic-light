import cv2
import time
import psutil
from ultralytics import YOLO
import numpy as np
import math
from collections import deque

# ---------------- Lane Detection Utils ---------------- #
last_left, last_right = None, None
left_history, right_history, angle_history = deque(maxlen=5), deque(maxlen=5), deque(maxlen=15)

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, [vertices], 255)
    return cv2.bitwise_and(img, mask)

def smooth_line(line, history):
    if line is not None:
        history.append(line)
    if not history:
        return None
    return np.mean(history, axis=0).astype(int).tolist()

def draw_lane_lines(img, left_line, right_line, color=(0, 255, 0)):
    global last_left, last_right
    if left_line is None: left_line = last_left
    else: last_left = left_line
    if right_line is None: right_line = last_right
    else: last_right = right_line
    if left_line is None or right_line is None: return img, None, None

    left_line, right_line = smooth_line(left_line, left_history), smooth_line(right_line, right_history)
    if left_line is None or right_line is None: return img, None, None

    line_img = np.zeros_like(img)
    pts = np.array([[(left_line[0], left_line[1]), (left_line[2], left_line[3]),
                     (right_line[2], right_line[3]), (right_line[0], right_line[1])]], dtype=np.int32)
    cv2.fillPoly(line_img, pts, color)
    return cv2.addWeighted(img, 0.8, line_img, 0.5, 0.0), left_line, right_line

def measure_curvature_and_angle(left_line, right_line, frame_width, frame_height):
    if left_line is None or right_line is None: return None, None, None
    xm_per_pix = 3.7 / 700
    lane_center = (left_line[0] + right_line[0]) // 2
    vehicle_center = frame_width // 2
    offset = (lane_center - vehicle_center) * xm_per_pix
    lookahead_y = 15.0
    steering_angle = -np.arctan2(offset, lookahead_y) * 180 / np.pi
    angle_history.append(steering_angle)
    steering_angle = float(np.median(angle_history))
    curvature = 5000.0 if abs(steering_angle) < 0.1 else abs(50 / (steering_angle / 45.0))
    curvature = np.clip(curvature, 100.0, 5000.0)
    return curvature, steering_angle, offset

def driving_direction(angle, curvature):
    if curvature is None: return "Go Straight"
    if curvature >= 725: dead_zone = 2.9
    elif 600 < curvature < 725: dead_zone = 3.66
    elif 450 < curvature <= 600: dead_zone = 2
    else: dead_zone = 4
    if -dead_zone <= angle <= dead_zone or curvature > 850: return "Go Straight"
    elif angle > dead_zone: return "Right"
    elif angle < -dead_zone: return "Left"
    return "Go Straight"

def draw_heading_line(img, angle, color=(0, 0, 255)):
    h, w = img.shape[:2]
    overlay = img.copy()
    angle_rad = math.radians(angle)
    x1, y1, x2, y2 = w // 2, h, int(w//2 + h/2 * math.tan(angle_rad)), int(h/2)
    cv2.line(overlay, (x1, y1), (x2, y2), color, 3)
    return cv2.addWeighted(img, 0.8, overlay, 1, 1)

def lane_pipeline(image):
    height, width = image.shape[:2]
    roi_vertices = np.array([(0, height), (width // 2, int(height * 0.55)), (width, height)], dtype=np.int32)
    gray, edges = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.Canny(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 100, 200)
    cropped = region_of_interest(edges, roi_vertices)
    lines = cv2.HoughLinesP(cropped, 2, np.pi/180, 50, minLineLength=30, maxLineGap=50)
    if lines is None: return image, None, None, None
    lx, ly, rx, ry = [], [], [], []
    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y2 - y1) / (x2 - x1) if (x2 - x1)!=0 else 0
            if abs(slope) < 0.5: continue
            (lx if slope<0 else rx).extend([x1,x2])
            (ly if slope<0 else ry).extend([y1,y2]) if slope<0 else ry.extend([y1,y2])
    min_y, max_y = int(height*0.6), height
    l_line = [int(np.poly1d(np.polyfit(ly,lx,1))(max_y)), max_y, int(np.poly1d(np.polyfit(ly,lx,1))(min_y)), min_y] if lx and ly else None
    r_line = [int(np.poly1d(np.polyfit(ry,rx,1))(max_y)), max_y, int(np.poly1d(np.polyfit(ry,rx,1))(min_y)), min_y] if rx and ry else None
    lane_img, l_line, r_line = draw_lane_lines(image, l_line, r_line)
    curvature, angle, offset = measure_curvature_and_angle(l_line, r_line, width, height)
    if curvature and angle is not None:
        direction = driving_direction(angle, curvature)
        if direction=="Go Straight": angle=0.0
        lane_img = draw_heading_line(lane_img, angle)
        cv2.putText(lane_img, f"Curvature: {curvature:.0f} m", (30,40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0),2)
        cv2.putText(lane_img, f"Steering: {angle:.1f}°", (30,80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0),2)
        cv2.putText(lane_img, f"Direction: {direction}", (30,120), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255),2)
    return lane_img, angle, curvature, offset

# ---------------- Main Combined Pipeline ---------------- #
def process_video():
    # Load YOLO models
    traffic_sign_model = YOLO(r"C:\Users\manas\Documents\traffic-sign-detection-using-yolov11-main\train10\weights\best.pt")
    traffic_light_model = YOLO(r"C:\Users\manas\Documents\traffic-sign-detection-using-yolov11-main\train6\weights\best.pt")

    input_path, output_path = r"D:\traffic-sign-detection-using-yolov11-main\WIN_20251002_15_12_21_Pro.mp4", r"combined_output1.mp4"
    cap = cv2.VideoCapture(1)
    if not cap.isOpened(): return print("Error: cannot open video")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (1280, 720))

    prev_time = time.time()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.resize(frame, (1280, 720))

        # Lane Detection
        lane_frame, steer, curv, offset = lane_pipeline(frame.copy())

        # Traffic Sign Detection
        results_signs = traffic_sign_model(lane_frame, verbose=False)
        for det in results_signs:
            for box in det.boxes:
                x1,y1,x2,y2 = map(int, box.xyxy[0])
                cv2.rectangle(lane_frame, (x1,y1), (x2,y2), (0,255,0), 2)
                cls = int(box.cls[0])
                cv2.putText(lane_frame, f"Sign {cls}", (x1,y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0),2)

        # Traffic Light Detection
        results_lights = traffic_light_model(lane_frame, verbose=False)
        for det in results_lights:
            for box in det.boxes:
                x1,y1,x2,y2 = map(int, box.xyxy[0])
                cv2.rectangle(lane_frame, (x1,y1), (x2,y2), (0,0,255), 2)
                cls = int(box.cls[0])
                cv2.putText(lane_frame, f"Light {cls}", (x1,y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)

        # FPS & RAM
        now, fps = time.time(), 1/(time.time()-prev_time)
        prev_time = now
        mem = psutil.Process().memory_info().rss / (1024*1024)
        cv2.putText(lane_frame, f"FPS: {fps:.1f}", (30,160), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(0,255,255),2)
        cv2.putText(lane_frame, f"RAM: {mem:.1f} MB", (30,200), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(0,255,255),2)

        out.write(lane_frame)
        cv2.imshow("Combined Lane + Traffic Sign + Light Detection", lane_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"): break

    cap.release(); out.release(); cv2.destroyAllWindows()
    print(f"[INFO] Saved combined output to {output_path}")

if __name__ == "__main__":
    process_video()
