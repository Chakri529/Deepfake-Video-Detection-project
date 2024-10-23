import cv2
import os
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array

# Function to detect faces using OpenCV's Haar cascades
def detect_faces(video_path, output_dir, face_cascade_path='haarcascade_frontalface_default.xml'):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + face_cascade_path)
    cap = cv2.VideoCapture(video_path)
    
    frame_count = 0
    faces = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected_faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in detected_faces:
            face = frame[y:y+h, x:x+w]
            face = cv2.resize(face, (224, 224))  # Resize face for model input
            faces.append(img_to_array(face))  # Store face as array
        
        frame_count += 1
    
    cap.release()
    cv2.destroyAllWindows()
    
    np.save(os.path.join(output_dir, 'faces.npy'), np.array(faces))

    return len(faces)
