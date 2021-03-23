from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dropout, Dense, BatchNormalization
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
import numpy as np
import tensorflow as tf
from math import ceil, floor
from scipy.linalg import sqrtm
from tensorflow_addons.image import rotate

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

def inception_score(p_yx, eps):
    p_y = np.expand_dims(p_yx.mean(axis=0), 0)
    kl_d = p_yx * (np.log(p_yx + eps) - np.log(p_y + eps))
    sum_kl_d = kl_d.sum(axis=1)
    avg_kl_d = np.mean(sum_kl_d)
    is_score = np.exp(avg_kl_d)
    return ceil(is_score)



def calculate_fid(real, fake):
    mu1, sigma1 = real.mean(axis=0), np.cov(real, rowvar=False)
    mu2, sigma2 = fake.mean(axis=0), np.cov(fake, rowvar=False)
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

def compute_score(X, Y, n_split=10, eps=1E-16):
    model = build_model(10)
    checkpoint_path = "./checkpoints/cifar"
    cls = tf.train.Checkpoint(model=model)
    cls_manager = tf.train.CheckpointManager(cls, checkpoint_path, max_to_keep=5)
    if cls_manager.latest_checkpoint:
        cls.restore(cls_manager.latest_checkpoint)
    prediction = model.predict(X)
    actual = model.predict(Y)
    fid = calculate_fid(prediction, actual)
    score_list = []
    n_part = floor(prediction.shape[0] / n_split)
    for i in range(n_split):
        ix_start, ix_end = i * n_part, (i + 1) * n_part
        subset_X = prediction[ix_start:ix_end, :]
        score_list.append(inception_score(subset_X, eps))
    is_avg, is_std = np.mean(score_list), np.std(score_list)
    ckpt_save_path = cls_manager.save()
    return fid, is_avg, is_std

def normalize(image):
  image = tf.cast(image, tf.float32)
  return image

if __name__ == '__main__':
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
    train_size = 50000
    test_size = 10000
    train_images = normalize(train_images)
    test_images = normalize(test_images)
    train_images, test_images = train_images / 255.0, test_images / 255.0
    batch_size = 60
    epochs = 30
    c_t = test_images
    c_l = test_labels
    for d in range(0, 100, 10):
        degree = np.radians(d)
        r_t = rotate(test_images, degree)
        c_t = np.concatenate((c_t, r_t))
        c_l = np.concatenate((c_l, test_labels))
    model = build_model(num_features=10)
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    filePath = "./cifar"
    checkpoint_path = "./checkpoints/" + filePath
    ckpt = tf.train.Checkpoint(model=model)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('classifier checkpoint restored!!')
    history = model.fit(
        c_t, c_l,
        epochs=epochs,
        shuffle=True,
        verbose=1)
    ckpt_save_path = ckpt_manager.save()
    print('Saving checkpoint for epoch {} at {}'.format(1,
                                                        ckpt_save_path))