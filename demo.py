import os
from PIL import Image
import numpy as np
import cv2
import torch
from torchvision import transforms
import math
from modules.emonet import Net
import mediapipe as mp


class Model():
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.data_transforms = transforms.Compose([
                                    transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])])
        self.labels = ['neutral', 'happy', 'sad', 'surprise', 'fear', 'disgust', 'anger', 'contempt']
        self.model = Net(pretrained=False)
        checkpoint = torch.load('./checkpoints/ckpt.pth', map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'],strict=True)
        self.model.to(self.device)
        self.model.eval()
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        mp_face_detection = mp.solutions.face_detection
        min_detection_confidence = 0.5
        #model_num = 0 #short range model
        model_num = 1 #full range model
        self.face_detector = mp_face_detection.FaceDetection(min_detection_confidence=min_detection_confidence, model_selection=model_num)
        
    def detect(self, img0):
        img = cv2.cvtColor(np.asarray(img0),cv2.COLOR_RGB2BGR)
        faces = self.face_cascade.detectMultiScale(img)
        return faces
        
    def image(self, path):
        img = Image.open(path).convert('RGB')
        faces = self.detect(img)

        if len(faces) == 0:
            return 'null'

        # single face detection
        x, y, w, h = faces[0]

        img = img.crop((x,y, x+w, y+h))
        img = self.data_transforms(img)
        img = img.view(1,3,224,224)
        img = img.to(self.device)

        with torch.no_grad():
            out, _, _ = self.model(img)
            _, pred = torch.max(out,1)
            index = int(pred)
            label = self.labels[index]
            return label

    def video(self, video_file):
        cap = cv2.VideoCapture(video_file)
        cnt = 0
        os.makedirs('result', exist_ok=True)

        while cap.isOpened():
            ret, frame = cap.read()

            if ret:
                frame1 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img0 = Image.fromarray(frame1).convert('RGB')
                results = self.face_detector.process(frame1)
                height, width, _ = frame1.shape

                if results.detections: 
                    data = results.detections[0]
                    x = round(data.location_data.relative_bounding_box.xmin * width)
                    y = round(data.location_data.relative_bounding_box.ymin * height) 
                    w = round(data.location_data.relative_bounding_box.width * width)
                    h = round(data.location_data.relative_bounding_box.height * height)

                    img = img0.crop((x,y, x+w, y+h))
                    img = self.data_transforms(img)
                    img = img.view(1,3,224,224)
                    img = img.to(self.device)
                    frame = cv2.rectangle(frame, (x, y), (x+w,y+h), (0, 255, 0), 3)

                    with torch.set_grad_enabled(False):
                        out, _, _ = self.model(img)
                        _, pred = torch.max(out,1)
                        index = int(pred)
                        label = self.labels[index]

                        font_scale = min(frame.shape[0], frame.shape[1]) * 2e-3
                        thickness = math.ceil(min(frame.shape[0], frame.shape[1]) * 1e-3)
                        cv2.putText(frame, label, (x, y-20), cv2.FONT_HERSHEY_PLAIN, font_scale, (0, 255, 0), thickness)
                cv2.imwrite(os.path.join('result/%06d.jpg'%(cnt)), frame)
                cnt = cnt + 1
            else:
                break
        cap.release()
        
if __name__ == "__main__":
    model = Model()   
    #emotion = model.image("temp.png")
    model.video('y2.mp4')
