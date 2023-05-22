import pickle
from flask import Flask, render_template, request,Response , send_file
import os
from random import random
from my_yolov6 import my_yolov6
import cv2
from imutils.video import VideoStream
from cv2 import VideoCapture
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

                file_extension = image.filename.rsplit('.', 1)[1].lower()
                print(file_extension)
               
                if file_extension == 'mp4':
                    cap = cv2.VideoCapture(path_to_save)
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        frame, det_len = yolov6_model.infer(frame)
                        cv2.imshow("Inference result", frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                        # _, buffer = cv2.imencode('.jpg', frame)
                        # yield (b'--frame\r\n'
                        #     b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n\r\n')  
                        # After the loop release the cap object
                    cap.release()
                    # Destroy all the windows
                    cv2.destroyAllWindows()
                else:
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


# def detect_video():
#     file = request.files['file']
#     video_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
#     file.save(video_path)
#     width = 640
#     height = 480
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
#     conf_thres = 0.25
#     iou_thres = 0.45
#     classes = None
#     agnostic_nms = False
#     max_det = 1000
#     cap = cv2.VideoCapture(video_path)
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         img_src, det_len = yolov6_model.infer(frame, conf_thres, iou_thres, classes, agnostic_nms, max_det)
#         _, buffer = cv2.imencode('.jpg', img_src)
#         yield (b'--frame\r\n'
#             b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n\r\n')  

# @app.route('/detect')
# def detect():
#         return Response(home_page(), mimetype='multipart/x-mixed-replace; boundary=frame')
    
@app.route("/webcam_feed")
def webcam_feed():
   # define a video capture object
    cap  = cv2.VideoCapture(0)
    width = 640
    height = 480
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    conf_thres = 0.25
    iou_thres = 0.45
    classes = None
    agnostic_nms = False
    max_det = 1000
    while(True):
        # Capture the video frame
        # by frame
        ret, frame = cap.read()
        frame = cv2.flip(frame,1)
        img_src, det_len = yolov6_model.infer(frame, conf_thres, iou_thres, classes, agnostic_nms, max_det)
        # for fire in frame:
        #     frame = yolov6_model.infer(frame)
        # Display the resulting frame  
        cv2.imshow("Inference result", img_src)
        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # After the loop release the cap object
    cap.release()
    # Destroy all the windows
    cv2.destroyAllWindows()
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)