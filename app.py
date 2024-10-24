from flask import Flask, request, jsonify, render_template, send_file
import torch
import cv2
import os
import pandas as pd

# 创建 Flask 应用
app = Flask(__name__)

# 加载你训练好的 YOLOv5 模型（此模型只检测大型卡车）
model = torch.hub.load('ultralytics/yolov5', 'custom', path='runs/train/exp34/weights/best.pt')

# 创建文件夹
os.makedirs("uploads", exist_ok=True)
os.makedirs("custom_images", exist_ok=True)
os.makedirs("excel_output", exist_ok=True)  # 保存Excel文件的目录
csv_file_path = os.path.join("excel_output", "detection_results.csv")

# 绘制边界框的函数
def draw_bounding_boxes(image, detections):
    for detection in detections:
        x1, y1, x2, y2, confidence, cls = detection
        if int(cls) == 0:  # 大卡车
            color = (0, 255, 0)  # 绿色
            label = "Large Truck"
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(image, f"{label} {confidence:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return image

# 检测卡车的独立函数，返回检测结果和大卡车数量
def detect_trucks(image_path):
    # 使用模型进行推理
    results = model(image_path)
    detections = results.xyxy[0].numpy()

    # 统计检测到的大卡车数量
    large_truck_count = len([det for det in detections if int(det[-1]) == 0])
    
    return detections, large_truck_count

# Flask 路由，用于检测大型卡车并返回带标注框的图片
@app.route('/detect', methods=['POST'])
def detect_trucks_api():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    # 获取上传的图像
    file = request.files['image']
    image_path = os.path.join("uploads", file.filename)
    file.save(image_path)

    # 调用卡车检测函数
    detections, large_truck_count = detect_trucks(image_path)

    # 读取图片并绘制边界框
    image = cv2.imread(image_path)
    image_with_boxes = draw_bounding_boxes(image, detections)

    # 保存带有标注框的图片
    output_image_path = os.path.join("custom_images", file.filename)
    cv2.imwrite(output_image_path, image_with_boxes)

    return jsonify({
        'num_large_trucks': large_truck_count,
        'output_image': output_image_path
    })

# Flask 路由，用于生成并返回Excel文件，保存所有检测到的结果
@app.route('/generate_excel', methods=['POST'])
def generate_excel_with_detection():
    if 'images' not in request.files:
        return jsonify({'error': 'No images uploaded'}), 400

    detection_results = []

    # 处理上传的多个图像
    for file in request.files.getlist('images'):
        image_path = os.path.join("uploads", file.filename)
        file.save(image_path)

        # 调用卡车检测函数
        detections, large_truck_count = detect_trucks(image_path)

        # 将检测结果保存到列表中
        detection_results.append({'image_name': file.filename, 'num_large_trucks': large_truck_count})

        # 读取图片并绘制边界框
        image = cv2.imread(image_path)
        image_with_boxes = draw_bounding_boxes(image, detections)

        # 保存带有标注框的图片
        output_image_path = os.path.join("custom_images", file.filename)
        cv2.imwrite(output_image_path, image_with_boxes)

    # 生成Excel文件
    df = pd.DataFrame(detection_results)
    excel_file_path = os.path.join("excel_output", "truck_detection_results.xlsx")
    df.to_excel(excel_file_path, index=False)

    return send_file(excel_file_path, as_attachment=True, attachment_filename='truck_detection_results.xlsx')

if __name__ == "__main__":
    # 监听本地或提供的端口
    port = int(os.environ.get('PORT', 8080))
    app.run(host="0.0.0.0", port=port)
