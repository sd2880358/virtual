import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import os
import scipy.io as sc
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.layers.experimental import preprocessing
import time
from IPython.display import clear_output
import math
from model import CVAE

def make_classifier():
    model = tf.keras.Sequential()
    model.add(layers.Dense(50, activation='relu'))
    model.add(layers.Dense(50, activation='relu'))
    model.add(layers.Dense(3, activation='relu'))
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  optimizer=tf.keras.optimizers.Adam(0.01),
                  metrics=['accuracy'])
    return model

cat_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

import math
def find_diff(model, x_1, x_2):
    mean_1, logvar_2 = model.encode(x_1)
    z_1 = model.reparameterize(mean_1, logvar_2).numpy()
    mean_2, logvar_2 = model.encode(x_2)
    z_2 = model.reparameterize(mean_2, logvar_2).numpy()
    diff = np.mean(np.abs(z_1 - z_2), axis=0)
    return diff

def split_label(model, data, labels, split=200):
    label_set = ["orientation", "x_axis", "y_axis"]
    tmp = []
    features = []
    for i in range(len(label_set)):
        l = len(labels.groupby(label_set[i]).count())
        for j in range(l):
            label_idx = labels[labels[label_set[i]] == j].index
            train_set = data[label_idx]
            np.random.shuffle(train_set)
            subgroups = math.ceil(train_set.shape[0]/split)
            for batch in range(subgroups):
                start = batch*split
                end = (batch+1)*split
                subset = train_set[start:end]
                s = int(len(subset)/2)
                x_1 = subset[:s]
                x_2 = subset[s:]
                diff = find_diff(model, x_1, x_2)
                features.append(diff)
                tmp.append(i)
    features = np.array(features, dtype="float32")
    tmp = np.array(tmp, dtype='int')
    return features, tmp


if __name__ == '__main__':
    model = CVAE(latent_dim=8, beta=4, shape=[64, 64, 1])
    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint.restore("checkpoints/4_25/beta_test/ckpt-31")
    dataset_zip = np.load('../dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')

    imgs = dataset_zip['imgs']
    imgs = np.reshape(imgs, [len(imgs), 64, 64, 1]).astype('float32')
    latents_values = dataset_zip['latents_values']
    latents_classes = dataset_zip['latents_classes']
    latents_classes = pd.DataFrame(latents_classes)
    latents_classes.columns = ["color", "shape", "scale", "orientation", "x_axis", "y_axis"]
    features, labels = split_label(model, imgs, latents_classes)
    train_features = features
    train_labels = labels
    classifer = make_classifier()
    history = classifer.fit(
        train_features, train_labels,
        validation_split=0.2,
        epochs=100, verbose=1)
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    test_results = {}
    hist.to_csv("./score/dis_matrix")