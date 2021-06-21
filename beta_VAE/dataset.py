import numpy as np
from tensorflow_addons.image import rotate
import pandas as pd
def preprocess_images(images):
  images = images.reshape((images.shape[0], 28, 28, 1)) / 255.
  return np.where(images > .5, 1.0, 0.0).astype('float32')

def divide_dataset(train_data, train_labels, sample_size):
  labels = pd.DataFrame({'labels': train_labels})
  dataset = []
  for i in range(0, 10):
    idx = labels[labels.labels == i].iloc[:sample_size].index
    train_images = train_data[idx]
    dataset.append(train_images)
  return np.array(dataset).reshape(10 * sample_size, 28, 28, 1)

def rotate_dataset(image, label, rotate_set):
  s = rotate_set[0]
  e = rotate_set[1]
  dataset = image
  labelset = label
  for degree in range(s+10, e+10, 10):
    d = np.radians(degree)
    r_image = rotate(image, d)
    dataset = np.concatenate([r_image, dataset])
    labelset = np.concatenate([labelset, label])
  return dataset, labelset


