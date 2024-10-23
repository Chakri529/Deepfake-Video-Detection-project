import tensorflow as tf
from tensorflow.keras.layers import GRU, Dense, Flatten, TimeDistributed, GlobalAveragePooling2D
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.models import Sequential

# Define the ResNet50V2 + GRU model
def build_model(input_shape=(None, 224, 224, 3)):
    base_model = ResNet50V2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False  # Freeze the base model

    model = Sequential()
    model.add(TimeDistributed(base_model, input_shape=input_shape))
    model.add(TimeDistributed(GlobalAveragePooling2D()))
    model.add(GRU(64, return_sequences=False))
    model.add(Dense(1, activation='sigmoid'))  # Binary classification (Real/Fake)

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Training function
def train_model(train_data, train_labels, val_data, val_labels, batch_size=32, epochs=10):
    model = build_model()
    
    history = model.fit(train_data, train_labels, validation_data=(val_data, val_labels),
                        epochs=epochs, batch_size=batch_size)
    
    model.save('models/deepfake_detection_model.h5')
    return history
