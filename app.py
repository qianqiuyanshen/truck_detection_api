from flask import Flask, request, jsonify, render_template, send_file
import torch
import cv2
import os
import pandas as pd

app = Flask(__name__)

# uolov5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='runs/train/exp34/weights/best.pt')

# folders
os.makedirs("uploads", exist_ok=True)
os.makedirs("custom_images", exist_ok=True)
os.makedirs("excel_output", exist_ok=True)  

detection_results = []

def draw_bounding_boxes(image, detections):
    for detection in detections:
        x1, y1, x2, y2, confidence, cls = detection
        if int(cls) == 0:  # 大卡车
            color = (0, 255, 0)  # 绿色
            label = "Large Truck"
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(image, f"{label} {confidence:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return image

# Flask 路由，用于检测大型卡车并保存带标注框的图片
@app.route('/detect', methods=['POST'])
def detect_trucks_api():
    if 'image' not in request.files:
        return jsonify({'error': 'No upload images'}), 400

    # 获取上传的图像
    file = request.files['image']
    image_path = os.path.join("uploads", file.filename)
    file.save(image_path)

    # 使用模型进行推理
    results = model(image_path)
    detections = results.xyxy[0].numpy()  # 获取推理结果

    # 读取图片并绘制边界框（只处理大型卡车）
    image = cv2.imread(image_path)
    image_with_boxes = draw_bounding_boxes(image, detections)

    # 保存带有标注框的图像
    output_image_path = os.path.join("custom_images", file.filename)
    cv2.imwrite(output_image_path, image_with_boxes)

    # 统计大型卡车的数量
    large_truck_count = len([det for det in detections if int(det[-1]) == 0])

    # 保存检测结果到列表
    detection_results.append({'image_name': file.filename, 'num_large_trucks': large_truck_count})

    return jsonify({
        'num_large_trucks': large_truck_count,
        'output_image': output_image_path
    })

# Flask 路由，用于生成并返回Excel文件
@app.route('/generate_excel', methods=['GET'])
def generate_excel():
    if len(detection_results) == 0:
        return jsonify({'error': 'No detection results available'})

    # 生成Excel文件
    df = pd.DataFrame(detection_results)
    excel_file_path = os.path.join("excel_output", "truck_detection_results.xlsx")
    df.to_excel(excel_file_path, index=False)

    # 返回生成的Excel文件
    return send_file(excel_file_path, as_attachment=True, attachment_filename='truck_detection_results.xlsx')

def send_images_request(folder_path, public_url):
    # 获取文件夹中所有图片文件
    for file_name in os.listdir(folder_path):
        # 过滤非图片文件
        if file_name.endswith(('.png', '.jpg', '.jpeg')):
            # 获取文件的完整路径
            file_path = os.path.join(folder_path, file_name)
            # 将文件复制到上传文件夹
            upload_path = os.path.join("uploads", file_name)
            with open(file_path, 'rb') as f:
                with open(upload_path, 'wb') as upload_file:
                    upload_file.write(f.read())
            
            # send POST to the server for the checking
            files = {'image': open(upload_path, 'rb')}
            # make sureURL 格式正确
            url = f"{public_url}/detect"
            try:
                response = requests.post(url, files=files)
                # 打印检测结果
                result = response.json()
                print(f"Results for {file_name}: Detected {result['num_large_trucks']} large trucks")
            except Exception as e:
                print(f"Error processing {file_name}: {e}")

if __name__ == "__main__":
    # 监听本地或提供的端口
    port = int(os.environ.get('PORT', 8080))
    app.run(host="0.0.0.0", port=port)
