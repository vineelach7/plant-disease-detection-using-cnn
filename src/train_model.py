import os
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint

# Data generators
train_path = 'data/train'
val_path = 'data/val'

train_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(train_path, target_size=(128,128), batch_size=32, class_mode='categorical')
val_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(val_path, target_size=(128,128), batch_size=32, class_mode='categorical')

# Model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Dropout(0.3),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(train_gen.num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
checkpoint = ModelCheckpoint('models/plant_disease_model.h5', save_best_only=True)
model.fit(train_gen, validation_data=val_gen, epochs=10, callbacks=[checkpoint])
