from model import Classifier
from dataset import preprocess_images, rotate_dataset, imbalance_sample

import numpy as np
import tensorflow as tf


(train_set, train_labels), (test_dataset, test_labels) = tf.keras.datasets.mnist.load_data()
train_images = preprocess_images(train_set)
test_images = preprocess_images(test_dataset)
classifier = Classifier(shape=(28, 28, 1))
filePath = "./imbalance_classification"
classifier_path = "./checkpoints/" + filePath
cls = tf.train.Checkpoint(classifier=classifier)
cls_manager = tf.train.CheckpointManager(cls, classifier_path, max_to_keep=5)
if cls_manager.latest_checkpoint:
    cls.restore(cls_manager.latest_checkpoint)
    print('classifier checkpoint restored!!')
irs = [4000, 2000, 1000, 750, 500, 350, 200, 100, 60, 40]

train_images, train_labels = imbalance_sample(train_images, train_labels, irs)
classifier.compile(optimizer='adam',
                   loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                   metrics=['accuracy'])
classifier.fit(train_images, train_labels, epochs=50, verbose=1, shuffle=True,
               batch_size=32,
               validation_data=(test_images,  test_labels))
test_loss, test_acc = classifier.evaluate(test_images,  test_labels, verbose=2)
print('\nTest accuracy:', test_acc)
ckpt_save_path = cls_manager.save()
print('Saving checkpoint for epoch {} at {}'.format(1,
                                                    ckpt_save_path))