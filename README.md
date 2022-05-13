# Implementation of a CNN for brain MRIs classification.

Author: alexcla99  
Version: 1.0.0

### Folder content:

```
+-+- preprocessed_dataset/                   # The folder containing the preprocessed dataset
|
+-+- results/                                # The folder containing both train and test results
| +-+- augmented_dataset/                    # The folder containing results for the augmented dataset
| +-+- normal_dataset/                       # The folder containing results for the normal dataset
|
+--- __init__.py                             # An empty file to make this directory being a Python library
+--- explore_results.py                      # A script to explore the results
+--- model.py                                # The model to be trained
+--- params.py                               # The params of the model and the train phase
+--- preprocess_to_numpy.py                  # A script to preprocess the inputs and store them into npy (Numpy) files
+--- README.md                               # This file
+--- requirements.txt                        # The Python libraries to be installed in order to run the project
+--- summarize_model.py                      # A script to print the model summary (Keras)
+--- test.py                                 # A script to test the model performances
+--- tf_config.py                            # A script to configure TensorFlow
+--- train.py                                # A script to train the model
+--- utils.py                                # Some utils
```

### Usage:

This library has been implemented and used with Python>=3.8.0

Requirements:
```Shell
pip3 install -r requirements
```

Preprocess the dataset:
```Shell
python3 preprocess_to_numpy
```
The data is selected from a folder called "normalized_dataset" (containing the raw dataset to be preprocessed) and is preprocessed into the "preprocessed_dataset" folder.

Summarize the model:
```Shell
python3 summarize_model.py
```

Train the model:
```Shell
python3 train.py <augment:bool> <debug:bool>
# Example: python3 train.py True False
```
The data to be used is selected from the "preprocessed_dataset" folder and the results are saved in the "results" folder.

Test the model:
```Shell
python3 test.py <augmented:bool> <debug:bool>
# Example: python3 test.py False True
```
The data to be used is selected from the "preprocessed_dataset" folder and the results are saved in the "results" folder.

### Many thanks to:

```Bib
@InProceedings{
	10.1007/978-3-030-98253-9_26,
	author="Saeed, Numan
	and Al Majzoub, Roba
	and Sobirov, Ikboljon
	and Yaqub, Mohammad",
	editor="Andrearczyk, Vincent
	and Oreiller, Valentin
	and Hatt, Mathieu
	and Depeursinge, Adrien",
	title="An Ensemble Approach for Patient Prognosis of Head and Neck Tumor Using Multimodal Data",
	booktitle="Head and Neck Tumor Segmentation and Outcome Prediction",
	year="2022",
	publisher="Springer International Publishing",
	address="Cham",
	pages="278--286",
	abstract="Accurate prognosis of a tumor can help doctors provide a proper course of treatment and, therefore, save the lives of many. Traditional machine learning algorithms have been eminently useful in crafting prognostic models in the last few decades. Recently, deep learning algorithms have shown significant improvement when developing diagnosis and prognosis solutions to different healthcare problems. However, most of these solutions rely solely on either imaging or clinical data. Utilizing patient tabular data such as demographics and patient medical history alongside imaging data in a multimodal approach to solve a prognosis task has started to gain more interest recently and has the potential to create more accurate solutions. The main issue when using clinical and imaging data to train a deep learning model is to decide on how to combine the information from these sources. We propose a multimodal network that ensembles deep multi-task logistic regression (MTLR), Cox proportional hazard (CoxPH) and CNN models to predict prognostic outcomes for patients with head and neck tumors using patients' clinical and imaging (CT and PET) data. Features from CT and PET scans are fused and then combined with patients' electronic health records for the prediction. The proposed model is trained and tested on 224 and 101 patient records respectively. Experimental results show that our proposed ensemble solution achieves a C-index of 0.72 on The HECKTOR test set that saved us the first place in prognosis task of the HECKTOR challenge. The full implementation based on PyTorch is available on https://github.com/numanai/BioMedIA-Hecktor2021.",
	isbn="978-3-030-98253-9"
}
```
