import numpy as np
import os
from config import config
import cv2
from sklearn.model_selection import train_test_split
import tensorflow as tf


class Dataset():
    def __init__(self, root_dir, img_size, mode='rgb'):
        super(Dataset, self).__init__()
        self.img_size = img_size
        self.train_path = root_dir + 'train1/img/'
        self.val_path = root_dir + 'valid1/img/'
        self.train_fid = os.listdir(self.train_path)
        self.valid_fid = os.listdir(self.val_path)
        self.train_fids = [self.train_path + fid for fid in self.train_fid]
        self.val_fids = [self.val_path + fid for fid in self.valid_fid]
        self.mode=mode

        self.train_data = tf.data.Dataset.from_generator(
            self.train_generator, output_types=(tf.string, tf.float32)).prefetch(tf.data.experimental.AUTOTUNE)
        self.valid_data = tf.data.Dataset.from_generator(
            self.valid_generator, output_types= (tf.float32, tf.string)).prefetch(tf.data.experimental.AUTOTUNE)
        self.length_train = len(self.train_fids)
        self.length_valid = len(self.val_fids)

    def train_generator(self):
        np.random.shuffle(self.train_fids)
        for start in range(0, len(self.train_fids), config.batch_size):
            x_batch = []
            fid_batch = []
            end = min(start + config.batch_size, len(self.train_fids))
            ids_train_batch = self.train_fids[start:end]
            np.random.shuffle(ids_train_batch)
            for fid in ids_train_batch:
                if self.mode == 'rgb':
                    rgb_image = cv2.imread(fid)
                elif self.mode == 'gray':
                    rgb_image = cv2.imread(fid, 0)
                    rgb_image = np.stack([rgb_image,rgb_image,rgb_image],-1)
                x_batch.append(rgb_image)
                fid_batch.append(fid)
            x_batch = np.array(x_batch, np.float32)
            yield fid_batch, x_batch

    def valid_generator(self):
        for start in range(0, len(self.val_fids), config.batch_size):
            x_batch = []
            end = min(start + config.batch_size, len(self.val_fids))
            ids_valid_batch = self.val_fids[start:end]
            np.random.shuffle(ids_valid_batch)
            for fid in ids_valid_batch:
                if self.mode == 'rgb':
                    rgb_image = cv2.imread(fid)
                elif self.mode == 'gray':
                    rgb_image = cv2.imread(fid, 0)
                x_batch.append(rgb_image)
            x_batch = np.array(x_batch, np.float32)
            yield x_batch, fid
