'''
import numpy as np
import tensorflow as tf

def preprocess_image(image_path, target_size=(224, 224)):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array
'''
# src/utils.py
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def create_data_generators(data_dir, img_size=(224, 224), batch_size=32):
    datagen_train = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        zoom_range=0.2,
        horizontal_flip=True
    )
    datagen_val_test = ImageDataGenerator(rescale=1./255)

    train = datagen_train.flow_from_directory(
        directory=f"{data_dir}/train", target_size=img_size, batch_size=batch_size, class_mode='categorical'
    )
    val = datagen_val_test.flow_from_directory(
        directory=f"{data_dir}/val", target_size=img_size, batch_size=batch_size, class_mode='categorical'
    )
    test = datagen_val_test.flow_from_directory(
        directory=f"{data_dir}/test", target_size=img_size, batch_size=batch_size, class_mode='categorical', shuffle=False
    )
    return train, val, test