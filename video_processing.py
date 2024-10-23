import cv2
import os

# Function to extract frames from video and save them
def extract_frames(video_path, output_dir):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_filename = os.path.join(output_dir, f'frame_{frame_count}.jpg')
        cv2.imwrite(frame_filename, frame)
        frame_count += 1
    
    cap.release()
    cv2.destroyAllWindows()

    return frame_count

# Function to load frames for model input
def load_frames(frame_dir):
    frames = []
    for frame_file in sorted(os.listdir(frame_dir)):
        frame_path = os.path.join(frame_dir, frame_file)
        frame = cv2.imread(frame_path)
        frame = cv2.resize(frame, (224, 224))  # Resize for model input
        frames.append(frame)
    
    return np.array(frames)
