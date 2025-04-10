import cv2
from ultralytics import YOLO
import numpy as np
import random

# 載入 YOLOv8 模型
model = YOLO("yolo12n.pt") 

# 讀取影片
cap = cv2.VideoCapture('boat.mp4')

# 設定輸出影片格式與參數
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 建立 VideoWriter 實例
out_diff = cv2.VideoWriter('diff_output.mp4', fourcc, fps, (frame_width, frame_height), isColor=False)
out_track = cv2.VideoWriter('tracking_output.mp4', fourcc, fps, (frame_width, frame_height), isColor=True)

# 讀取第一幀
ret, previous_frame = cap.read()
if not ret:
    print("無法讀取影片或攝影機")
    cap.release()
    cv2.destroyAllWindows()
    exit()

# 前處理（灰階 + 高斯模糊）
previous_frame_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
previous_frame_gray = cv2.GaussianBlur(previous_frame_gray, (21, 21), 0)

# 紀錄軌跡：目前的與所有歷史軌跡
trajectory_points = []
trajectory_groups = []

# 紀錄上一次相對於垂直線的位置：'left' or 'right'
last_position = None

# 進出計數
count_in = 0
count_out = 0

# 幀數計數
frame_count = 0
color = (0, 255, 0)

while True:
    ret, current_frame = cap.read()
    if not ret:
        break

    frame_height, frame_width = current_frame.shape[:2]
    line_x = frame_width // 2
    cv2.line(current_frame, (line_x, 0), (line_x, frame_height), (255, 0, 0), 2)

    # 影格差分
    gray_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)
    frame_diff = cv2.absdiff(previous_frame_gray, gray_frame)
    _, thresh = cv2.threshold(frame_diff, 20, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_x, min_y = float('inf'), float('inf')
    max_x, max_y = 0, 0
    area_threshold = 10

    for contour in contours:
        if cv2.contourArea(contour) < area_threshold:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        min_x = min(min_x, x)
        min_y = min(min_y, y)
        max_x = max(max_x, x + w)
        max_y = max(max_y, y + h)

    # YOLO 偵測
    results = model(current_frame)
    is_ship = False

    if min_x < max_x and min_y < max_y:
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls.item())
                cls_name = model.names[cls_id]
                conf = box.conf.item()

                if (cls_name == 'boat' or cls_name == 'ship') and conf > 0.05:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    x_overlap = max(0, min(max_x, x2) - max(min_x, x1))
                    y_overlap = max(0, min(max_y, y2) - max(min_y, y1))
                    overlap_area = x_overlap * y_overlap
                    motion_area = (max_x - min_x) * (max_y - min_y)
                    if overlap_area > 0.3 * motion_area:
                        is_ship = True
                        break

    # 判斷軌跡與位置
    frame_count += 1
    if frame_count > 300:
        if len(trajectory_points) > 1:
            trajectory_groups.append({
                'points': trajectory_points,
                'color': color
            })
        trajectory_points = []
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        frame_count = 0

    if min_x < max_x and min_y < max_y:
        if is_ship:
            frame_count = 0
            cv2.rectangle(current_frame, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)
            center_x = (min_x + max_x) // 2
            center_y = (min_y + max_y) // 2
            trajectory_points.append((center_x, center_y))

            current_position = 'left' if center_x < line_x else 'right'
            if last_position and last_position != current_position:
                if last_position == 'left' and current_position == 'right':
                    count_in += 1
                    print("船隻從左側穿越到右側，count_in =", count_in)
                elif last_position == 'right' and current_position == 'left':
                    count_out += 1
                    print("船隻從右側穿越到左側，count_out =", count_out)
            last_position = current_position
        else:
            cv2.rectangle(current_frame, (min_x, min_y), (max_x, max_y), (0, 0, 255), 2)

    # 繪製歷史軌跡
    for group in trajectory_groups:
        for i in range(1, len(group['points'])):
            cv2.line(current_frame, group['points'][i - 1], group['points'][i], group['color'], 2)
    # 當前船軌跡
    for i in range(1, len(trajectory_points)):
        cv2.line(current_frame, trajectory_points[i - 1], trajectory_points[i], color, 2)

    # 顯示計數文字
    text_info = f"In(L->R): {count_in}   Out(R->L): {count_out}"
    cv2.putText(current_frame, text_info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # 更新前幀
    previous_frame_gray = gray_frame

    # 顯示與寫入影片
    cv2.imshow("Frame Difference (Thresh)", thresh)
    cv2.imshow("Ship Detection with Tracking", current_frame)
    out_diff.write(thresh)
    out_track.write(current_frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# 收尾
cap.release()
out_diff.release()
out_track.release()
cv2.destroyAllWindows()
