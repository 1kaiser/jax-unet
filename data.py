import glob, os
import tensorflow as tf
import numpy as np

PATH = 'datasets/'


def read_data():
    x_files = [f for f in glob.glob(PATH + "train/*.jpg", recursive=True)]
    y_files = [f for f in glob.glob(PATH + "mask/*.jpg", recursive=True)]

    def read_image(x_filename, y_filename):
        x_image_string = tf.io.read_file(x_filename)
        y_image_string = tf.io.read_file(y_filename)

        x_image_decoded = tf.image.decode_jpeg(x_image_string, channels=3)
        y_image_decoded = tf.image.decode_jpeg(y_image_string, channels=1)

        # y_image_decoded = tf.image.rgb_to_grayscale(y_image_decoded)

        x_image_resized = tf.image.resize(x_image_decoded, [512, 512])
        y_image_resized = tf.image.resize(y_image_decoded, [512, 512])

        return x_image_resized, y_image_resized

    dataset = tf.data.Dataset.from_tensor_slices((x_files, y_files))

    dataset = dataset.map(read_image).shuffle(1000).batch(12)

    return dataset
