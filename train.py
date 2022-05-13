"""
Author: alexcla99
Version: 1.0.0
Title: Test implementation of a CNN for brain MRIs classification (train phase)
"""

# Imports
from utils import info, load_datasets, data_expansion
from tf_config import tf_configure
from model import get_model
from params import run_params
from tensorflow import keras
from tensorflow.keras import layers
from distutils.util import strtobool
import numpy as np
import tensorflow as tf
import sys, os

AUTOTUNE = tf.data.experimental.AUTOTUNE

def prepare_dataset(ds:tf.data.Dataset, n_shuffle:int, augment:bool=False) -> tf.data.Dataset:
    """Prepare the dataset by augmenting / enpanding / suffling / prefetching it."""
    ds = ds.map(data_expansion)
    if augment == True:
        data_augmentation = tf.keras.Sequential([
            layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
            layers.experimental.preprocessing.RandomRotation(.2),
        ])
        ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)
    ds = ds.shuffle(n_shuffle)
    ds = ds.batch(run_params["batch_size"])
    return ds.prefetch(buffer_size=AUTOTUNE)

if __name__ == "__main__":
    """Load the entire pipeline and train the model."""
    if len(sys.argv) != 3:
        print("Usage: python3 train.py <augment:bool> <debug:bool>")
        print("Example: python3 train.py True False")
    else:
        augment = strtobool(str(sys.argv[1]))
        debug = strtobool(str(sys.argv[2]))
        try:
            # Starting a fresh session
            tf.keras.backend.clear_session()
            tf_configure()
            # Building the train / val datasets
            info("Loading and splitting datasets:")
            x_train, x_val, x_test, y_train, y_val, y_test = load_datasets()
            if debug == True:                # DEBUG
                x_train = x_train[:2]        # Train with 2 MRIs
                x_val = x_val[:2]            # Validate with 2 MRIs
                y_train = y_train[:2]        # Train with 2 labels
                y_val = y_val[:2]            # Validate with 2 labels
                run_params["epochs"] = 2     # Train on 2 epochs
                run_params["patience"] = 2   # Same for patience
                run_params["batch_size"] = 1 # Train on the smallest batch size
            print("Number of samples in train/val are %d/%d" % (x_train.shape[0], x_val.shape[0]))
            train_loader = tf.data.Dataset.from_tensor_slices((x_train, y_train))
            val_loader = tf.data.Dataset.from_tensor_slices((x_val, y_val))
            # Preparing the train / val datasets
            train_dataset = prepare_dataset(train_loader, len(x_train), augment)
            val_dataset = prepare_dataset(val_loader, len(x_val))
            # Defining the model and distributing data along all available GPUs
            info("Defining the model")
            if debug == True:                # DEBUG
                model = get_model()          # Data distribution is disabled
            else:
                mirrored_strategy = tf.distribute.MirroredStrategy()
                with mirrored_strategy.scope():
                    model = get_model()
            # Compiling the model
            info("Compiling the model")
            optimizer = keras.optimizers.Adam(learning_rate=run_params["lr"])
            model.compile(
                loss=run_params["loss"],
                optimizer=optimizer, 
                metrics="acc"
            )
            dest_path = "results"
            dest_path = os.path.join(dest_path, "augmented_dataset" if augment == True else "normal_dataset")
            # Monitors (best parameters / early stopping / learning rate reducer)
            checkpoint_cb = keras.callbacks.ModelCheckpoint(
                os.path.join(dest_path, "3d_image_classifier.h5"), save_best_only=True
            )
            early_stopping_cb = keras.callbacks.EarlyStopping(
                monitor="val_acc", min_delta=1e-4, patience=run_params["patience"], verbose=1
            )
            lr_reducer = keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=np.sqrt(.1), cooldown=0, patience=10, min_lr=0.5e-6, verbose=1
            )
            # Training the model
            info("Training the model:")
            model.fit(
                train_dataset,
                validation_data=val_dataset,
                epochs=run_params["epochs"],
                shuffle=True,
                verbose=2,
                callbacks=[checkpoint_cb, early_stopping_cb, lr_reducer]
            )
            # Computing model performances
            info("Computing model performances")
            for metric in ["acc", "loss"]:
                np.save(os.path.join(dest_path, metric + ".npy"), model.history.history[metric], allow_pickle=False)
                np.save(os.path.join(dest_path, "val_" + metric + ".npy"), model.history.history["val_" + metric], allow_pickle=False)
            # End of the program
            info("Done")
        except:
            print(str(sys.exc_info()[1]))