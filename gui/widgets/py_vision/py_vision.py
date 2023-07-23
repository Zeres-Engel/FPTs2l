import cv2
from PySide6.QtCore import QTimer
from PySide6.QtGui import QImage, QPixmap, Qt
from PySide6.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout
# import gui.mediapipe as mp
import mediapipe as mp
import numpy as np

mp_holistic = mp.solutions.holistic  # Holistic model
mp_drawing = mp.solutions.drawing_utils  # Drawing utilities

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False  # Image is no longer writeable
    results = model.process(image)  # Make prediction
    image.flags.writeable = True  # Image is now writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # COLOR COVERSION RGB 2 BGR
    return image, results

def draw_landmarks(image, results):

    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                                mp_drawing.DrawingSpec(color=(0, 176, 14), thickness=1, circle_radius=1), 
                                mp_drawing.DrawingSpec(color=(255, 127, 39), thickness=1, circle_radius=1)
                                )
    # Vẽ pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(77, 77, 77), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=1)
                                )
    # Vẽ left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(255, 127, 39), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(166, 129, 76), thickness=2, circle_radius=2)
                                )
    # Vẽ right hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(255, 127, 39), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(166, 129, 76), thickness=2, circle_radius=2)
                                )




def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(34 * 4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)

    # Calculate coordinate differences
    pose_bone = np.array([pose[i + 4] - pose[i] if (i + 1) % 4 != 0 else pose[i] for i in range(len(pose) - 4)])
    face_bone = np.array([face[i + 3] - face[i] if (i + 1) % 3 != 0 else face[i] for i in range(len(face) - 3)])
    lh_bone = np.array([lh[i + 3] - lh[i] if (i + 1) % 3 != 0 else lh[i] for i in range(len(lh) - 3)])
    rh_bone = np.array([rh[i + 3] - rh[i] if (i + 1) % 3 != 0 else rh[i] for i in range(len(rh) - 3)])

    # Concatenate the differences to form the feature vector
    joint = np.concatenate([pose, face, lh, rh])
    bone = np.concatenate([pose_bone, face_bone, lh_bone, rh_bone])
    return joint, bone


import time

class ModelFunction(QWidget):
    def __init__(self, model_joint, model_bone, model_joint_motion, model_bone_motion):
        super().__init__()
        
        self.model_joint = model_joint
        self.model_bone = model_bone
        self.model_joint_motion = model_joint_motion
        self.model_bone_motion = model_bone_motion
        
        self.joint_motions = []
        self.bone_motions = []
        self.joint_window = []
        self.bone_window = []
        
        self.J = None
        self.B = None
        self.JM = None
        self.BM = None
        
        self.modelJ = None
        self.modelB = None
        self.modelJM = None
        self.modelBM = None
        
        self.cJ =False
        self.cB = False
        self.cJM = False

        self.actions = np.array(['goodbye', 'hello', 'hi', 'like', 'love'])
        
        self.action = None

        self.predictions = []

        self.cap = cv2.VideoCapture(0)

        # Khởi tạo Holistic model
        self.holistic = mp_holistic.Holistic()

        # Kiểm tra và thiết lập tốc độ khung hình tối đa của webcam
        if self.cap.isOpened():
            supported_fps = [30, 60]  # Danh sách tốc độ khung hình hỗ trợ
            max_fps = max(supported_fps)  # Lấy tốc độ khung hình tối đa
            # Thiết lập tốc độ khung hình của webcam
            self.cap.set(cv2.CAP_PROP_FPS, max_fps)

        # Khung hình hiển thị video từ webcam
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)

        # Tạo QVBoxLayout và thêm video_label vào layout
        layout = QVBoxLayout()
        layout.addWidget(self.video_label)

        # Thiết lập layout cho cửa sổ chính
        self.setLayout(layout)

        self.frame_count = 0
        self.start_time = time.time()

        # Tạo một QTimer để gọi hàm update_frame mỗi 16ms (khoảng 60 FPS)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(16)
        
    def update_frame(self):
        # Đọc khung hình từ webcam
        ret, frame = self.cap.read()

        if ret:
            # Gọi hàm mediapipe_detection để nhận diện và vẽ khung xương
            image, results = mediapipe_detection(frame, self.holistic)
            draw_landmarks(image, results)
            
            joint, bone = extract_keypoints(results)
            
            if self.joint_window:
                self.JM = joint - self.joint_window[-1]
                self.BM = bone - self.bone_window[-1]
                
                self.joint_motions.append(self.JM)
                self.joint_motions = self.joint_motions[-30:]
                
                self.bone_motions.append(self.BM)
                self.bone_motions = self.bone_motions[-30:]
            
            self.joint_window.append(joint)
            self.joint_window = self.joint_window[-30:]
            
            self.bone_window.append(bone)
            self.bone_window = self.bone_window[-30:]
            if len(self.joint_motions) == 30:
                if self.cJ == False:
                    # Dự đoán J và xuất kết quả
                    self.modalJ = self.model_joint.predict(np.expand_dims(self.joint_window, axis=0))[0]
                    self.cJ = True
                elif self.cJM == False:
                    # Dự đoán JM và xuất kết quả
                    self.modalJM = self.model_joint_motion.predict(np.expand_dims(self.joint_motions, axis=0))[0]
                    self.cJM = True
                elif self.cB == False:
                    # Dự đoán B và xuất kết quả
                    self.modalB = self.model_bone.predict(np.expand_dims(self.bone_window, axis=0))[0]
                    self.cB = True
                else:
                    # Dự đoán BM và xuất kết quả
                    self.modalBM = self.model_bone_motion.predict(np.expand_dims(self.bone_motions, axis=0))[0]
                    self.cJ = False
                    self.cB = False
                    self.cJM = False
                weights = np.array([0.25, 0.25, 0.25, 0.25])
                ensemble_prob = np.dot(weights, [self.modalJ, self.modalB, self.modalJM, self.modalBM])
                self.action = self.actions[np.argmax(ensemble_prob)]
                self.display_image(image, self.action)
            

    def display_image(self, image, action):
        if (action not in self.predictions):
            self.predictions.append(action)
            if len(self.predictions) == 5:
                self.predictions.pop(0)
        else:
            cv2.putText(image, action, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        result = ""
        for predict in self.predictions:
            result += f"{predict} "
        cv2.putText(image, result, (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 1, (0x21, 0x6f, 0xf2), 2)

        # Chuyển đổi khung hình từ OpenCV BGR format sang QImage
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        q_image = QImage(
            image.data, image.shape[1], image.shape[0], QImage.Format_RGB888
        )
        # Chuyển đổi QImage thành QPixmap để hiển thị trên QLabel
        pixmap = QPixmap.fromImage(q_image)

        # Thay đổi kích thước pixmap để phù hợp với kích thước của QLabel
        pixmap = pixmap.scaled(
            self.video_label.size(), Qt.AspectRatioMode.KeepAspectRatio
        )
        # Hiển thị pixmap trên QLabel
        self.video_label.setPixmap(pixmap)
        






