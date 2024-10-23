import numpy as np
from tensorflow.keras.models import load_model

# Function to load the trained model and test it on new data
def evaluate_model(test_data, test_labels):
    model = load_model('models/deepfake_detection_model.h5')
    loss, accuracy = model.evaluate(test_data, test_labels)
    
    print(f'Test Accuracy: {accuracy * 100:.2f}%')
    return accuracy

# Function to make predictions
def predict_on_video(video_data):
    model = load_model('models/deepfake_detection_model.h5')
    predictions = model.predict(video_data)
    return predictions
