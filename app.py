from flask import Flask, render_template, Response
import cv2, os, face_recognition
import numpy as np
import trainer
import detectors
import generator

app = Flask(__name__)

def gen_register_frame(id):
  offset = 50
  i = 0

  while True:
    ret, frame = cam.read()
    gray = cv2.cvtColor(frame ,cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100), flags=cv2.CASCADE_SCALE_IMAGE)
    for(x,y,w,h) in faces:
        i = i + 1
        cv2.imwrite("dataSet/face-"+ id +'.'+ str(i) + ".jpg", gray[y-offset:y+h+offset,x-offset:x+w+offset])
        cv2.rectangle(frame, (x-50,y-50),(x+w+50,y+h+50),(225,0,0),2)
        cv2.waitKey(100)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    if i>20:
        cam.release()
        cv2.destroyAllWindows()
        break
  
  trainer.train()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/face-register')
def face_register():
  return render_template('face_register.html')

@app.route('/video_detector_feed')
def video_detector_feed():
    return Response(detectors.detect(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_register_feed')
def video_register_feed():
    return Response(generator.generate("12"), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__=='__main__':
    app.run(debug=True)