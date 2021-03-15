import os
import numpy as np


def load_celeba(path):
    data = np.load(os.path.join(path, "data.npy"))
    data = data.astype(float)
    data = data / 255.0  # -1 ~ 1
    return data


if __name__ == "__main__":
    test = load_celeba("../CelebA/")
    print(len(test))