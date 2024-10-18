from flask import Flask, request, jsonify, render_template
import torch
import cv2
import os

# 创建 Flask 应用
app = Flask(__name__)

# 加载你训练好的 YOLOv5 模型（此模型只检测大型卡车）
model = torch.hub.load('ultralytics/yolov5', 'custom', path='runs/train/exp34/weights/best.pt')

# 创建文件夹
os.makedirs("uploads", exist_ok=True)
os.makedirs("custom_images", exist_ok=True)

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

# 显示上传页面的路由
@app.route('/upload', methods=['GET'])
def upload_page():
    return render_template('upload.html')

# 用于检测大型卡车的路由
@app.route('/detect', methods=['POST'])
def detect_trucks_api():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    # 获取上传的图像
    file = request.files['image']
    image_path = os.path.join("uploads", file.filename)
    file.save(image_path)

    # 使用模型进行推理
    results = model(image_path)
    detections = results.xyxy[0].numpy()

    # 读取图片并绘制边界框
    image = cv2.imread(image_path)
    image_with_boxes = draw_bounding_boxes(image, detections)

    # 保存带有标注框的图像
    output_image_path = os.path.join("custom_images", file.filename)
    cv2.imwrite(output_image_path, image_with_boxes)

    # 返回检测到的大型卡车数量和带边界框的图片路径
    large_truck_count = len([det for det in detections if int(det[-1]) == 0])
    return jsonify({
        'num_large_trucks': large_truck_count,
        'output_image': output_image_path
    })

if __name__ == "__main__":
    # 监听本地或提供的端口
    port = int(os.environ.get('PORT', 8080))
    app.run(host="0.0.0.0", port=port)
