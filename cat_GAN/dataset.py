from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dropout, Dense, BatchNormalization
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from celebA import CelebA
import numpy as np
import tensorflow as tf
from math import ceil, floor
from scipy.linalg import sqrtm
import os

def load_celeba(path):
    data = np.load(os.path.join(path, "data.npy"))
    data = data.astype(float)
    data = (data / 127.5) - 1
    return data


if __name__ == '__main__':
    dataset = load_celeba("../CelebA/")
    batch_size = 60
    epochs = 20
    celeba = CelebA(drop_features=[
    'Attractive',
    'Pale_Skin',
    'Blurry',
    ])
    train_datagen = ImageDataGenerator(rotation_range=100,
                                       rescale=1./255,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True,
                                    fill_mode='nearest')
    valid_datagen = ImageDataGenerator(rescale=1./255)
    train_split = celeba.split('training'  , drop_zero=False)
    valid_split = celeba.split('validation', drop_zero=False)
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_split,
        directory=celeba.images_folder,
        x_col='image_id',
        y_col=celeba.features_name,
        target_size=(32, 32),
        batch_size=batch_size,
        class_mode='other'
    )
    valid_generator = valid_datagen.flow_from_dataframe(
        dataframe=valid_split,
        directory=celeba.images_folder,
        x_col='image_id',
        y_col=celeba.features_name,
        target_size=(32, 32),
        batch_size=batch_size,
        class_mode='other'
    )
    model = build_model(num_features=celeba.num_features)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics='binary_accuracy')
    filePath = "./celebA"
    checkpoint_path = "./checkpoints/" + filePath
    ckpt = tf.train.Checkpoint(model=model)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('classifier checkpoint restored!!')
    print(len(train_generator))
    history = model.fit_generator(
        train_generator,
        epochs=epochs,
        steps_per_epoch=len(train_generator),
        validation_data=valid_generator,
        validation_steps=len(valid_generator),
        max_queue_size=1,
        shuffle=True,
        verbose=1)
    ckpt_save_path = ckpt_manager.save()
    print('Saving checkpoint for epoch {} at {}'.format(1,
                                                        ckpt_save_path))