import pickle
from flask import Flask, render_template, request,Response
import os
from random import random
from my_yolov6 import my_yolov6
import cv2
from imutils.video import VideoStream
import time

yolov6_model = my_yolov6("weights/last_ckpt.pt", "cpu", "data/mydataset.yaml", 640, True)

# Khởi tạo Flask
app = Flask(__name__)
app.config['CORS_HEADERS'] = 'Content-Type'
app.config['UPLOAD_FOLDER'] = "static"

# Hàm xử lý request
@app.route("/", methods=['GET', 'POST'])
def home_page():
    # Nếu là POST (gửi file)
    if request.method == "POST":
         try:
            # Lấy file gửi lên
            image = request.files['file']
            if image:
                # Lưu file
                path_to_save = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
                print("Save = ", path_to_save)
                image.save(path_to_save)

                # Convert image to dest size tensor
                frame = cv2.imread(path_to_save)

                frame, no_object = yolov6_model.infer(frame)
                if no_object > 0:
                    cv2.imwrite(path_to_save, frame)

                    # Trả về kết quả
                    return render_template("index.html", user_image = image.filename , rand = str(random()),msg="Tải file lên thành công")
            else:
                # Nếu không có file thì yêu cầu tải file
                return render_template('index.html', msg='Hãy chọn file để tải lên')
         except Exception as ex:
            # Nếu lỗi thì thông báo
            print(ex)
            return render_template('index.html', msg='Không nhận diện được vật thể')

    else:
        # Nếu là GET thì hiển thị giao diện upload
        return render_template('index.html')

@app.route("/webcam_feed")
def webcam_feed():
    video = VideoStream(src=0).start()
    while True:
        frame = video.read()
        frame = cv2.flip(frame, 1)
        cv2.imshow("Intrusion Warning", frame)
        return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)