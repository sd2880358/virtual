import zipfile
from tqdm import tqdm
import os
import cv2
import numpy as np
from PIL import Image
import glob


if __name__ == "__main__":
    ZIP_PATH = "img_align_celeba.zip"
    CROP_WIDTH = 128
    CROP_HEIGHT = 128
    OUTPUT_WIDTH = 32
    OTUPUT_HEIGHT = 32
    SAVE_PATH = "data.npy"
    image_files = "img_algin_celeba"

    archive = zipfile.ZipFile(ZIP_PATH, "r")
    print(os.path.exists(ZIP_PATH))
    images = []
    for image_file in glob.glob(image_files):
        image = archive.read(image_file)
        image = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
        height, width, _ = image.shape
        s_height = (height - CROP_HEIGHT) / 2
        s_width = (width - CROP_WIDTH) / 2
        image = image[s_height: s_height + CROP_HEIGHT,
                      s_width: s_width + CROP_WIDTH,
                      :]
        image = cv2.resize(image, (OUTPUT_WIDTH, OTUPUT_HEIGHT))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(image)
    images = np.stack(images, axis=0)
    print(images.shape)
    np.save(SAVE_PATH, images)