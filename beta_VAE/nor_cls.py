from model import Classifier
from dataset import preprocess_images, rotate_dataset

import numpy as np
import tensorflow as tf


(train_set, train_labels), (test_dataset, test_labels) = tf.keras.datasets.mnist.load_data()
train_images = preprocess_images(train_set)
test_images = preprocess_images(test_dataset)
classifier = Classifier(shape=(28, 28, 1))
filePath = "./base_line_classification"
classifier_path = "./checkpoints/" + filePath
cls = tf.train.Checkpoint(classifier=classifier)
cls_manager = tf.train.CheckpointManager(cls, classifier_path, max_to_keep=5)
if cls_manager.latest_checkpoint:
    cls.restore(cls_manager.latest_checkpoint)
    print('classifier checkpoint restored!!')
partial_range_dataset = train_images[np.where(train_labels!=3)]
partial_range_labels = train_labels[np.where(train_labels!=3)]
partial_range_datase, partial_range_labels = rotate_dataset(train_images, train_labels[0, 180])
full_range_dataset = train_images[np.where(train_labels==3)]
full_range_labels = train_labels[np.where(train_labels==3)]
full_range_dataset, full_range_labels = rotate_dataset(full_range_dataset, full_range_labels, [0, 360])
train_images = np.concatenate([partial_range_dataset, full_range_dataset])
train_labels = np.concatenate([partial_range_labels, full_range_labels])
test_images, test_labels = rotate_dataset(test_images, test_labels, [0, 360])
classifier.compile(optimizer='adam',
                   loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                   metrics=['accuracy'])
classifier.fit(train_images, train_labels, epochs=50, verbose=1, shuffle=True,
               validation_data=(test_images,  test_labels))
test_loss, test_acc = classifier.evaluate(test_images,  test_labels, verbose=2)
print('\nTest accuracy:', test_acc)
ckpt_save_path = cls_manager.save()
print('Saving checkpoint for epoch {} at {}'.format(1,
                                                    ckpt_save_path))