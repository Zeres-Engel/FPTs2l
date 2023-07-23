import cv2
import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model


from sklearn.model_selection import train_test_split


    
def extract_joint(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(34*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)

    # Concatenate the differences to form the feature vector
    joint = np.concatenate([pose, face, lh, rh])

    return joint

def extract_bone(joint):
    # Extract pose from joint
    pose_size = 132
    pose = joint[:pose_size]

    # Extract face from joint
    face_size = 1404
    face = joint[pose_size:pose_size + face_size]

    # Extract left hand from joint
    lh_size = 63
    lh = joint[pose_size + face_size:pose_size + face_size + lh_size]

    # Extract right hand from joint
    rh_size = 63
    rh = joint[pose_size + face_size + lh_size:pose_size + face_size + lh_size + rh_size]
    
    # Calculate coordinate differences
    pose_bone = np.array([pose[i+4]-pose[i] if (i+1) % 4 != 0 else pose[i] for i in range(len(pose)-4)])
    face_bone = np.array([face[i+3]-face[i] if (i+1) % 3 != 0 else face[i] for i in range(len(face)-3)])
    lh_bone = np.array([lh[i+3]-lh[i] if (i+1) % 3 != 0 else lh[i] for i in range(len(lh)-3)])
    rh_bone = np.array([rh[i+3]-rh[i] if (i+1) % 3 != 0 else rh[i] for i in range(len(rh)-3)])
    
    bone = np.concatenate([pose_bone, face_bone, lh_bone, rh_bone])
    return bone

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(34*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)

    # Calculate coordinate differences
    pose_bone = np.array([pose[i+4]-pose[i] if (i+1) % 4 != 0 else pose[i] for i in range(len(pose)-4)])
    face_bone = np.array([face[i+3]-face[i] if (i+1) % 3 != 0 else face[i] for i in range(len(face)-3)])
    lh_bone = np.array([lh[i+3]-lh[i] if (i+1) % 3 != 0 else lh[i] for i in range(len(lh)-3)])
    rh_bone = np.array([rh[i+3]-rh[i] if (i+1) % 3 != 0 else rh[i] for i in range(len(rh)-3)])

    # Concatenate the differences to form the feature vector
    joint = np.concatenate([pose, face, lh, rh])
    bone = np.concatenate([pose_bone, face_bone, lh_bone, rh_bone])
    return joint, bone


# Load and preprocess data
actions = os.listdir('./Data')

label_map = {label: num for num, label in enumerate(actions)}

sequences, labels = [], []

for action in actions:
    numbers = os.listdir(os.path.join('./Data', action))
    for num in numbers:
        try:
            bone_motions = []
            bone_window = extract_bone(np.load(os.path.join('./Data', action, num, f"0.npy")))
            for frame in range(1, 31):
                bone = extract_bone(np.load(os.path.join('./Data', action, num, f"{frame}.npy")))
                bone_motions.append(bone - bone_window)
                bone_window = bone
            sequences.append(bone_motions)
            labels.append(label_map[action])
        except:
            continue

X = np.array(sequences)
y = to_categorical(labels).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

print(np.array(sequences).shape)
print(np.array(labels).shape)
print(X.shape)
print(y_train.shape)

log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, np.array(sequences).shape[2])))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(len(actions), activation='softmax'))


if os.path.isfile('bone_motion.h5'):
  model = load_model('bone_motion.h5')
  print('pretrained')


model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# Định nghĩa Early Stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Thêm Early Stopping callback vào quá trình huấn luyện
model.fit(X_train, y_train, epochs=1000, use_multiprocessing=True, shuffle=True, validation_split=0.2, callbacks=[early_stopping])

# Save the model
model.save('bone_motion.h5')
