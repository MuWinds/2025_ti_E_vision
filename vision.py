import cv2
import numpy as np
from collections import deque

# 初始化摄像头
cap = cv2.VideoCapture(1) 
if not cap.isOpened():
    print("无法打开摄像头")
    exit()

# 创建参数控制窗口
cv2.namedWindow('Parameters')
cv2.resizeWindow('Parameters', 600, 300)

# 初始参数值
cv2.createTrackbar('Thresh_BlockSize', 'Parameters', 11, 51, lambda x: None) # BlockSize必须是奇数
cv2.createTrackbar('Thresh_C', 'Parameters', 2, 50, lambda x: None)
cv2.createTrackbar('black_vmax', 'Parameters', 50, 255, lambda x: None)
cv2.createTrackbar('hough_dp_x10', 'Parameters', 12, 50, lambda x: None)
cv2.createTrackbar('hough_param1', 'Parameters', 50, 255, lambda x: None)
cv2.createTrackbar('hough_param2', 'Parameters', 30, 100, lambda x: None)
cv2.createTrackbar('min_radius', 'Parameters', 10, 200, lambda x: None)
cv2.createTrackbar('max_radius', 'Parameters', 50, 500, lambda x: None)
cv2.createTrackbar('min_dist', 'Parameters', 5, 100, lambda x: None)

# 中心点缓存队列
centers_cache = deque(maxlen=4)  # 保存最近10个检测到的中心点

def order_points(pts):
    """按左上->右上->右下->左下的顺序给四个点排序"""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]      # 左上
    rect[2] = pts[np.argmax(s)]      # 右下
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]   # 右上
    rect[3] = pts[np.argmax(diff)]   # 左下
    return rect

def four_point_transform(img, pts):
    """执行四点透视变换，强制输出为 A4 比例"""
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    if maxWidth > maxHeight:
        new_width = maxWidth
        new_height = int(maxWidth / 1.414)
    else:
        new_height = maxHeight
        new_width = int(maxHeight * 1.414)

    dst = np.array([
        [0, 0],
        [new_width - 1, 0],
        [new_width - 1, new_height - 1],
        [0, new_height - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, M, (new_width, new_height))
    return warped, M

def calculate_centroid(points):
    """计算点集的中心点"""
    if not points:
        return None
    x = sum([p[0] for p in points]) / len(points)
    y = sum([p[1] for p in points]) / len(points)
    return (int(x), int(y))

while True:
    # 获取滑块参数
    black_vmax = cv2.getTrackbarPos('black_vmax', 'Parameters')
    dp = cv2.getTrackbarPos('hough_dp_x10', 'Parameters') / 10.0
    param1 = cv2.getTrackbarPos('hough_param1', 'Parameters')
    param2 = cv2.getTrackbarPos('hough_param2', 'Parameters')
    min_radius = cv2.getTrackbarPos('min_radius', 'Parameters')
    max_radius = cv2.getTrackbarPos('max_radius', 'Parameters')
    min_dist = cv2.getTrackbarPos('min_dist', 'Parameters')

    # 读取视频帧
    ret, frame = cap.read()
    if not ret:
        print("无法获取帧")
        break
    
    img_display = frame.copy()
    determined_center = None  # 本帧确定的中心点

    # 1. 识别包含同心圆的矩形区域 (A4纸)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred_frame = cv2.GaussianBlur(gray, (15, 5), 0)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, black_vmax])
    # --- 获取自适应阈值的参数 ---
    block_size = cv2.getTrackbarPos('Thresh_BlockSize', 'Parameters')
    C = cv2.getTrackbarPos('Thresh_C', 'Parameters')
    # 确保 block_size 是奇数
    if block_size % 2 == 0:
        block_size += 1
    if block_size <= 1: # block_size必须大于1
        block_size = 3
    # mask = cv2.inRange(hsv, lower_black, upper_black)
    mask = cv2.adaptiveThreshold(
        blurred_frame, 
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV,  # 先用INV试试，如果不行就换成 cv2.THRESH_BINARY
        block_size, 
        C
    )
    
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    cv2.imshow('mask', mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 5000:
            continue
            
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        
        if len(approx) == 4:
            cv2.drawContours(img_display, [approx], -1, (0, 255, 0), 3)

            try:
                warped_roi, M = four_point_transform(frame, approx.reshape(4, 2))
                if warped_roi.size == 0:
                    continue

                # 图像预处理
                gray_roi = cv2.cvtColor(warped_roi, cv2.COLOR_BGR2GRAY)
                blurred_roi = cv2.GaussianBlur(gray_roi, (5, 5), 0)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                enhanced_roi = clahe.apply(blurred_roi)
                final_roi_for_hough = cv2.medianBlur(enhanced_roi, 5)
                
                cv2.imshow('Enhanced Gray ROI', final_roi_for_hough)
                
                # 霍夫圆检测
                if param1 == 0: param1 = 1
                if param2 == 0: param2 = 1
                
                circles = cv2.HoughCircles(
                    final_roi_for_hough,
                    cv2.HOUGH_GRADIENT,
                    dp=dp,
                    minDist=min_dist,
                    param1=param1,
                    param2=param2,
                    minRadius=min_radius,
                    maxRadius=max_radius
                )

                warped_display = warped_roi.copy()
                if circles is not None:
                    circles = np.uint16(np.around(circles))
                    
                    # 圆心分组算法
                    main_centers = []
                    for i in circles[0, :]:
                        center = (i[0], i[1])
                        radius = i[2]
                        
                        # 如果当前圆圈半径大于阈值（避免噪声），添加到备选
                        if radius > min_radius:
                            cv2.circle(warped_display, center, radius, (0, 255, 0), 2)
                            cv2.circle(warped_display, center, 2, (0, 0, 255), 3)
                            main_centers.append(center)
                    
                    # 计算所有备选圆心的中心点
                    if main_centers:
                        avg_center = calculate_centroid(main_centers)
                        cv2.circle(warped_display, avg_center, 5, (255, 0, 0), -1)  # 蓝色中心点
                        
                        # 转换回原始图像坐标
                        M_inv = np.linalg.inv(M)
                        roi_center_array = np.array([[[avg_center[0], avg_center[1]]]], dtype=np.float32)
                        original_point_array = cv2.perspectiveTransform(roi_center_array, M_inv)
                        original_point = tuple(original_point_array[0,0].astype(int))
                        
                        # 添加到中心点缓存
                        centers_cache.append(original_point)
                        determined_center = original_point

                cv2.imshow('Warped ROI with Circles', warped_display)
                
            except Exception as e:
                # print(f"处理错误: {e}")
                pass
            break  # 只处理第一个检测到的矩形区域

    # 使用缓存中心点计算平滑后的中心（即使当前帧没有检测到中心点）
    if centers_cache:
        # 计算缓存中心点的平均值
        stable_center = calculate_centroid(centers_cache)
        
        # 在原始图像上绘制稳定中心点
        cv2.circle(img_display, stable_center, 10, (0, 0, 255), -1)  # 红色实心圆点
        cv2.circle(img_display, stable_center, 20, (0, 0, 255), 2)   # 红色空心圆圈
        
        # 显示中心点坐标
        cv2.putText(img_display, f"Center: {stable_center}", 
                   (stable_center[0] + 30, stable_center[1] - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.imshow('Detection', img_display)
    
    if cv2.waitKey(1) & 0xFF == 27:  # ESC退出
        break

cap.release()
cv2.destroyAllWindows()