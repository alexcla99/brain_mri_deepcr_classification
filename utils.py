import numpy as np
import tensorflow as tf
import os

def info(i:str) -> None:
    """Display an info about the process being run."""
    print("[INFO] " + str(i))

def load_datasets() -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """Load the datasets into tensors."""
    src_path = "preprocessed_dataset"
    x_train = np.load(os.path.join(src_path, "x_train.npy"))
    x_val = np.load(os.path.join(src_path, "x_val.npy"))
    x_test = np.load(os.path.join(src_path, "x_test.npy"))
    y_train = np.load(os.path.join(src_path, "y_train.npy"))
    y_val = np.load(os.path.join(src_path, "y_val.npy"))
    y_test = np.load(os.path.join(src_path, "y_test.npy"))
    return x_train, x_val, x_test, y_train, y_val, y_test

def data_expansion(volume:np.ndarray, label:np.ndarray) -> (np.ndarray, np.ndarray):
    """Process by adding a channel."""
    volume = tf.expand_dims(volume, axis=3)
    return volume, label