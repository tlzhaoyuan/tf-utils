import collections
import itertools

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def train_val_test_split(X, y, val_size, test_size, shuffle, verbose=False):
    """Split data into train/val/test datasets.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, shuffle=shuffle)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_size, stratify=y_train, shuffle=shuffle)
    if verbose:
        class_counts = dict(collections.Counter(y))
        print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
        print(f"X_val: {X_val.shape}, y_val: {y_val.shape}")
        print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")
        print(f"X_train[0]: {X_train[0]}")
        print(f"y_train[0]: {y_train[0]}")
        print(f"Classes: {class_counts}")
    return X_train, X_val, X_test, y_train, y_val, y_test

def standardize(X_train, X_val, X_test, verbose=False):
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    if verbose:
        # Check
        print(
            f"standardized_X_train: "
            f"mean: {np.mean(X_train, axis=0)[0]}, "
            f"std: {np.std(X_train, axis=0)[0]}")
        print(
            f"standardized_X_val: "
            f"mean: {np.mean(X_val, axis=0)[0]}, "
            f"std: {np.std(X_val, axis=0)[0]}")
        print(
            f"standardized_X_test: "
            f"mean: {np.mean(X_test, axis=0)[0]}, "
            f"std: {np.std(X_test, axis=0)[0]}")
    return X_train, X_val, X_test




def pad_sequence(seq, max_len):
    padded = pad_sequences(seq, max_len)
    return padded