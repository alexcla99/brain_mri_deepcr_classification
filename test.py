"""
Author: alexcla99
Version: 1.0.0
Title: Test implementation of a CNN for brain MRIs classification (test phase)
"""

# Imports
from utils import info, load_datasets
from tf_config import tf_configure
from model import get_model
from distutils.util import strtobool
import tensorflow as tf
import numpy as np
import sys, os

THRESHOLD = .5 # TODO test different values

if __name__ == "__main__":
    """Load the model with its best parameters and test it."""
    if len(sys.argv) != 3:
        print("Usage: python3 test.py <augmented:bool> <debug:bool>")
        print("Example: python3 test.py False True")
    else:
        augmented = strtobool(str(sys.argv[1]))
        debug = strtobool(str(sys.argv[2]))
        try:
            # Starting a fresh session
            tf.keras.backend.clear_session()
            tf_configure()
            # Building the test dataset
            info("Loading and splitting the dataset:")
            x_train, x_val, x_test, y_train, y_val, y_test = load_datasets()
            if debug == True:       # DEBUG
                x_test = x_test[:2] # Test with 2 MRIs
                y_test = y_test[:2] # Test with 2 labels
            print("Number of samples in test are %d" % (x_test.shape[0]))
            # Defining the model
            info("Defining the model")
            model = get_model()
            # Testing the model
            info("Testing the model")
            dest_path = "results"
            dest_path = os.path.join(dest_path, "augmented_dataset" if augmented == True else "normal_dataset")
            model.load_weights(os.path.join(dest_path, "3d_image_classifier.h5"))
            predictions = list()
            for e in x_test:
                prediction = model.predict(np.expand_dims(e, axis=0))[0]
                predictions.append(prediction)
            normalize = lambda x: 0. if x < THRESHOLD else 1. # Normalizatino since the model is trained with few data 
                                                              # and the output is given by a sigmoid activation function
            np.save(os.path.join(dest_path, "x_test_predictions.npy"), np.asarray([normalize(x) for x in predictions], dtype=np.float32), allow_pickle=False)
            np.save(os.path.join(dest_path, "y_test.npy"), y_test, allow_pickle=False)
            # End of the program
            info("Done")
        except:
            print(str(sys.exc_info()[1]))