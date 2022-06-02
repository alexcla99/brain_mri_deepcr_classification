from sklearn.model_selection import train_test_split
from glob import glob
import nibabel as nib
import numpy as np
import sys, os

RANDOM_SEED = 1337
TRAIN_RATIO = .6
VAL_RATIO = .2
TEST_RATIO = .2

def load_nifti_file(filepath:str) -> np.ndarray:
    """Compute the matrix of a nifti image."""
    volume = nib.load(filepath).dataobj
    volume = np.asarray(volume, dtype=np.float32)
    volume /= volume.max()
    return volume

def load_datasets() -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """Load the different datasets and build the train / val / test subsets."""
    src_path = "normalized_dataset"
    coma_path = glob(os.path.join(src_path, "coma", "*.nii"))
    control_path = glob(os.path.join(src_path, "control", "*.nii"))
    coma_data = np.array([load_nifti_file(e) for e in coma_path])
    control_data = np.array([load_nifti_file(e) for e in control_path])
    coma_labels = np.array([1. for _ in range(len(coma_data))])
    control_labels = np.array([0. for _ in range(len(control_data))])
    print("Coma MRIs: " + str(len(coma_data)))
    print("Control MRIs: " + str(len(control_data)))
    coma_x_train, coma_x_test, coma_y_train, coma_y_test = train_test_split(coma_data, coma_labels, test_size=1-TRAIN_RATIO, shuffle=False)
    coma_x_val, coma_x_test, coma_y_val, coma_y_test = train_test_split(coma_x_test, coma_y_test, test_size=TEST_RATIO/(TEST_RATIO+VAL_RATIO), shuffle=False)
    control_x_train, control_x_test, control_y_train, control_y_test = train_test_split(control_data, control_labels, test_size=1-TRAIN_RATIO, shuffle=False)
    control_x_val, control_x_test, control_y_val, control_y_test = train_test_split(control_x_test, control_y_test, test_size=TEST_RATIO/(TEST_RATIO+VAL_RATIO), shuffle=False)
    print("Coma train MRIs: " + str(len(coma_x_train)))
    print("Coma test MRIs: " + str(len(coma_x_test)))
    print("Coma validation MRIs: " + str(len(coma_x_val)))
    print("Control train MRIs: " + str(len(control_x_train)))
    print("Control test MRIs: " + str(len(control_x_test)))
    print("Control validation MRIs: " + str(len(control_x_val)))
    x_train = np.concatenate((coma_x_train, control_x_train), axis=0)
    x_val = np.concatenate((coma_x_val, control_x_val), axis=0)
    x_test = np.concatenate((coma_x_test, control_x_test), axis=0)
    y_train = np.concatenate((coma_y_train, control_y_train), axis=0)
    y_val = np.concatenate((coma_y_val, control_y_val), axis=0)
    y_test = np.concatenate((coma_y_test, control_y_test), axis=0)
    np.random.seed(RANDOM_SEED)
    np.random.shuffle(x_train)
    np.random.shuffle(x_val)
    np.random.shuffle(x_test)
    np.random.shuffle(y_train)
    np.random.shuffle(y_val)
    np.random.shuffle(y_test)
    print("Train MRIs: " + str(len(x_train)))
    print("Test MRIs: " + str(len(x_test)))
    print("Validation MRIs: " + str(len(x_val)))
    return x_train, x_val, x_test, y_train, y_val, y_test

if __name__ == "__main__":
    """Preprocess the dataset into numpy objects."""
    if len(sys.argv) != 1:
        print("Usage: python3 preprocess_to_numpy.py")
    else:
        try:
            dest_path = "preprocessed_dataset"
            x_train, x_val, x_test, y_train, y_val, y_test = load_datasets()
            np.save(os.path.join(dest_path, "x_train.npy"), x_train, allow_pickle=False)
            np.save(os.path.join(dest_path, "x_val.npy"), x_val, allow_pickle=False)
            np.save(os.path.join(dest_path, "x_test.npy"), x_test, allow_pickle=False)
            np.save(os.path.join(dest_path, "y_train.npy"), y_train, allow_pickle=False)
            np.save(os.path.join(dest_path, "y_val.npy"), y_val, allow_pickle=False)
            np.save(os.path.join(dest_path, "y_test.npy"), y_test, allow_pickle=False)
            print("Preprocessing done")
        except:
            print(str(sys.exc_info()[1]))
