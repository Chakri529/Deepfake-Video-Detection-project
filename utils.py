import numpy as np

# Normalization utility
def normalize_images(image_array):
    return image_array.astype('float32') / 255.0

# Splitting data into training and validation sets
def split_data(data, labels, validation_split=0.2):
    split_index = int(len(data) * (1 - validation_split))
    train_data, val_data = data[:split_index], data[split_index:]
    train_labels, val_labels = labels[:split_index], labels[split_index:]
    return train_data, train_labels, val_data, val_labels
