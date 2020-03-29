import pandas as pd
import tensorflow as tf
import numpy as np


def load_agnews(data_file):
    df = pd.read_csv(data_file)
    return df


class MNISTLoader():
    def __init__(self):
        mnist = tf.keras.datasets.mnist
        (self.train_data, self.train_label), (self.test_data, self.test_label) = mnist.load_data()
        self.train_data = pd.np.expand_dims(self.train_data.astype(np.float32) / 255.0, axis=-1)
        self.test_data = np.expand_dims(self.test_data.astype(np.float32) / 255.0, axis=-1)
        self.train_label = self.train_label.astype(np.int32)
        self.test_label = self.test_label.astype(np.int32)

    def get_batch(self, batch_size):
        indices = np.random.randint(0, self.train_data.shape[0], batch_size)
        return self.train_data[indices, :], self.train_label[indices]


if __name__ == '__main__':
    DATA_FILE = '../data/news.csv'
    news = load_agnews(DATA_FILE)
    X = news['title'].values
    y = news['category'].values
