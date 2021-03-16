from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dropout, Dense, BatchNormalization
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from load_data import load_celeba
from celebA import CelebA

import tensorflow as tf
def build_model(num_features):
    base = MobileNetV2(input_shape=(32, 32, 3),
                       weights=None,
                       include_top=False,
                       pooling='avg')  # GlobalAveragePooling 2D

    x = base.output
    x = Dense(1536, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    top = Dense(num_features, activation='sigmoid')(x)

    return Model(inputs=base.input, outputs=top)

if __name__ == '__main__':
    dataset = load_celeba("../CelebA/")
    batch_size = 60
    epochs = 30
    celeba = CelebA(drop_features=[
    'Attractive',
    'Pale_Skin',
    'Blurry',
    ])
    train_datagen = ImageDataGenerator(rotation_range=20,
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
    train_set = dataset[train_split.index]
    train_datagen.fit(train_set)
    train_generator = train_datagen.flow(
        train_set,
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
    model.compile(loss='cosine_proximity',
                  optimizer='adadelta',
                  metrics='binary_accuracy')

    filePath = "./celebA"
    checkpoint_path = "./checkpoints/" + filePath
    ckpt = tf.train.Checkpoint(model=model)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
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