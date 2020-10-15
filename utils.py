import os
import numpy as np
import tensorflow as tf
from PIL import Image


def create_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
        print(f'Directory {dir} createrd')
    else:
        print(f'Directory {dir} already exists')  

    return dir


def imsave(image, path):
    image = (image + 1) * 127.5
    image = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=255.0)
    image = Image.fromarray(np.array(image).astype(np.uint8).squeeze())
    image.save(path)