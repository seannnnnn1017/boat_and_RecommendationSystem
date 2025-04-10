import cv2
from ultralytics import YOLO
import numpy as np

# 載入 YOLOv8 模型
model = YOLO("yolo12n.pt")  # 使用 YOLOv8 nano 模型，您也可以使用其他大小的模型

# 讀取影片 (或攝影機: 可以把 '0824.mp4' 換成 0, 1, ...等索引)
cap = cv2.VideoCapture('0824.mp4')

ret, previous_frame = cap.read()
if not ret:
    print("無法讀取影片或攝影機")
    cap.release()
    cv2.destroyAllWindows()
    exit()

# 前處理（灰階 + 高斯模糊）
previous_frame_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
previous_frame_gray = cv2.GaussianBlur(previous_frame_gray, (21, 21), 0)

# 用來儲存每幀 bounding box 的中心點軌跡 (可視化用)
trajectory_points = []

# 紀錄上一次相對於垂直線的位置：'left' or 'right'
last_position = None

# 計算左→右 / 右→左 次數
count_in = 0   # 例如：左→右 (可以視為 "進")
count_out = 0  # 例如：右→左 (可以視為 "出")

while True:
    ret, current_frame = cap.read()
    if not ret:
        break

    # 取得當前畫面的寬度與高度
    frame_height, frame_width = current_frame.shape[:2]

    # (1) 設定垂直線的位置 (畫面正中央)
    line_x = frame_width // 2

    # (2) 在畫面上畫出這條「垂直線」(藍色)
    cv2.line(current_frame, (line_x, 0), (line_x, frame_height), (255, 0, 0), 2)

    #===============================
    #     影格差分, 找出物件
    #===============================
    gray_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)

    frame_diff = cv2.absdiff(previous_frame_gray, gray_frame)
    # 閾值化：大於 20 視為有差異
    _, thresh = cv2.threshold(frame_diff, 20, 255, cv2.THRESH_BINARY)

    # 尋找輪廓 (contours)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 初始化用於統計「大 bounding box」的座標 (給一組極端值)
    min_x, min_y = float('inf'), float('inf')
    max_x, max_y = 0, 0

    # 設定一個面積閾值，過小的區域視為雜訊
    area_threshold = 10

    for contour in contours:
        if cv2.contourArea(contour) < area_threshold:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        # 更新「整體的大 bounding box」的邊界
        if x < min_x:
            min_x = x
        if y < min_y:
            min_y = y
        if x + w > max_x:
            max_x = x + w
        if y + h > max_y:
            max_y = y + h

    #===============================
    #    使用 YOLO 進行物體偵測
    #===============================
    # 執行 YOLO 檢測
    results = model(current_frame)
    
    # 檢測結果
    is_ship = False
    motion_box = None
    
    # 只有當我們有一個有效的運動方塊時才進行船隻檢測
    if min_x < max_x and min_y < max_y:
        motion_box = (min_x, min_y, max_x, max_y)
        
        # 從YOLO結果中篩選出船隻 (類別 id=8 是 'boat' 在 COCO 資料集中)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # 取得分類結果
                cls_id = int(box.cls.item())
                cls_name = model.names[cls_id]
                conf = box.conf.item()
                
                # 檢查是否為船 (boat 或 ship)
                if (cls_name == 'boat' or cls_name == 'ship') and conf > 0.05:
                    # 取得 YOLO 的邊界框
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # 計算與運動檢測的框重疊
                    # 計算重疊區域
                    x_overlap = max(0, min(max_x, x2) - max(min_x, x1))
                    y_overlap = max(0, min(max_y, y2) - max(min_y, y1))
                    overlap_area = x_overlap * y_overlap
                    
                    # 計算運動檢測框的面積
                    motion_area = (max_x - min_x) * (max_y - min_y)
                    
                    # 如果重疊面積大於運動檢測框面積的一定比例，判定為船
                    if overlap_area > 0.3 * motion_area:
                        is_ship = True
                        break
    
    #===============================
    #    取得物體中心 & 判斷位置
    #===============================
    if min_x < max_x and min_y < max_y:
        # 只有是船才處理
        if is_ship:
            # 在畫面上畫出「藍色的 bounding box」表示船
            cv2.rectangle(current_frame, (min_x, min_y), (max_x, max_y), (255, 0, 0), 2)

            # 計算這個大型 bounding box 的中心點
            center_x = (min_x + max_x) // 2
            center_y = (min_y + max_y) // 2

            # 把中心點存進 trajectory_points (給下方連線用)
            trajectory_points.append((center_x, center_y))

            # 判斷「現在」在 left 還是 right
            if center_x < line_x:
                current_position = 'left'
            else:
                current_position = 'right'

            # 假如上一幀有位置、且發生位置改變 => 代表剛好穿越了垂直線
            if last_position is not None and last_position != current_position:
                if last_position == 'left' and current_position == 'right':
                    count_in += 1   # 左→右
                    print("船隻從左側穿越到右側，count_in =", count_in)
                elif last_position == 'right' and current_position == 'left':
                    count_out += 1  # 右→左
                    print("船隻從右側穿越到左側，count_out =", count_out)

            # 更新 last_position
            last_position = current_position
        else:
            # 不是船，畫綠色框（可選）
            cv2.rectangle(current_frame, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)
            # 不是船，不更新 last_position，保持原值

    #========================================
    #  在畫面上把所有中心點連線 (軌跡) - 只有船才有
    #========================================
    for i in range(1, len(trajectory_points)):
        cv2.line(current_frame,
                 trajectory_points[i - 1],  # 上一幀的中心
                 trajectory_points[i],      # 當前幀的中心
                 (0, 0, 255), 2)            # 用紅線(示意)

    #========================================
    #  顯示目前「進 / 出」累計 (可自行排版)
    #========================================
    text_info = f"In(L->R): {count_in}   Out(R->L): {count_out}"
    cv2.putText(current_frame, text_info, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # 更新「前一幀」
    previous_frame_gray = gray_frame

    # 顯示畫面
    cv2.imshow("Frame Difference (Thresh)", thresh)
    cv2.imshow("Ship Detection with Tracking", current_frame)

    # 按下 'q' 離開
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()