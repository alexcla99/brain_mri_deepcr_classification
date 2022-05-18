import numpy as np
import os

def print_results(src:str) -> None:
	"""Print the different obtained reults."""
	acc = np.load(os.path.join(src, "acc.npy"))
	loss = np.load(os.path.join(src, "loss.npy"))
	val_acc = np.load(os.path.join(src, "val_acc.npy"))
	val_loss = np.load(os.path.join(src, "val_loss.npy"))
	x_test_predictions = np.load(os.path.join(src, "x_test_predictions.npy"))
	y_test = np.load(os.path.join(src, "y_test.npy"))
	print("Results for the " + src)
	print("> Accuracy: " + np.array2string(acc))
	print("> Validation accuracy: " + np.array2string(val_acc))
	print("> Loss: " + np.array2string(loss))
	print("> Validation loss: " + np.array2string(val_loss))
	print("> Predictions: " + np.array2string(x_test_predictions.flatten()))
	print("> Ground truths: " + np.array2string(y_test))
	print()

# Exploring the normal_dataset results
print_results(os.path.join("results", "normal_dataset"))

# Exploring the augmented_dataset results
print_results(os.path.join("results", "augmented_dataset"))