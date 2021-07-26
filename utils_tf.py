import os
import numpy as np
from PIL import Image

from multiprocessing import Pool
from keras.preprocessing.image import img_to_array


class ImgLoader:
    def __init__(self, image_folder, shape=(224, 224)):
        self.shape = shape
        self.image_folder = image_folder

    def __call__(self, image_path):
        im = Image.open(os.path.join(self.image_folder, image_path)).resize(self.shape)
        if im.mode != 'RGB':
            im = im.convert("RGB")
        im = img_to_array(im)
        im = np.expand_dims(im, axis=0)
        return im


def load_images(image_folder, img_list, shape=(224, 224), process=4):
    pool = Pool(process)
    proc = ImgLoader(image_folder, shape=shape)

    images = [result for result in pool.imap(proc, img_list)]
    pool.close()
    pool.join()

    images = np.concatenate(np.array(images, dtype=np.float32), axis=0)
    return images
