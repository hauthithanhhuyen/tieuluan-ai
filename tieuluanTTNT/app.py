from flask import Flask, render_template, request, Response
import cv2
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Import thư viện FER xử lý AI
try:
    from fer import FER
except ImportError:
    from fer.fer import FER 

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Khởi tạo AI (mtcnn=False để camera chạy nhanh hơn)
detector = FER(mtcnn=False) 

EMOTIONS_VN = {
    "happy": "Vui 😄", "sad": "Buồn bã 😢", 
    "angry": "Tức giận 😡", "surprise": "Ngạc nhiên 😲",
    "neutral": "Bình thường 😐", "fear": "Sợ hãi 😨", "disgust": "Ghê tởm 🤮"
}

# Hàm vẽ chữ Tiếng Việt có dấu
def draw_vn_text(img, text, position):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    # Sử dụng font Arial mặc định của Windows
    font = ImageFont.truetype("arial.ttf", 30)
    draw.text(position, text, font=font, fill=(0, 255, 0))
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def generate_frames():
    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()
        if not success: break
        results = detector.detect_emotions(frame)
        for res in results:
            (x, y, w, h) = res['box']
            dominant = max(res['emotions'], key=res['emotions'].get)
            label = EMOTIONS_VN.get(dominant, "Bình thường 😐")
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            frame = draw_vn_text(frame, label, (x, y - 35))
        
        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/', methods=['GET', 'POST'])
def index():
    label_vn = None
    image_path = None
    if request.method == 'POST':
        file = request.files.get('file')
        if file:
            path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(path)
            img = cv2.imread(path)
            res = detector.detect_emotions(img)
            if res:
                dom = max(res[0]['emotions'], key=res[0]['emotions'].get)
                label_vn = EMOTIONS_VN.get(dom, "Bình thường 😐")
                box = res[0]['box']
                cv2.rectangle(img, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), (0, 255, 0), 2)
                img = draw_vn_text(img, label_vn, (box[0], box[1]-40))
                cv2.imwrite(path, img)
                image_path = path
    return render_template('index.html', label=label_vn, image_path=image_path)

if __name__ == '__main__':
    app.run(debug=True)