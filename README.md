# Road Detection

This repository is part of the Wetland project in collaboration with CCNB-Innov, focused on detecting logging roads. The main goal is to automate the detection and mapping of logging roads using artificial intelligence models trained on satellite imagery.

The algorithms can be trained using our directories below as steps in order.

1. Create the data using **Data Creation**
	- Code in this folder will generate both training and validation images into their respective folders
2. Create the training scripts **Training**
	- This folder will generate the grid search scripts to launch on ComputeCanada
3. Evaluate the results using **Results**
	- Based on the select models from **Training**, adjust the prediction code and predict images.

## Directory Structure

### Data Creation
This folder contains scripts and resources for creating the dataset used for training and validation.
For more information, refer to [here](Data_creation/Readme.md)

- **Folders:**
  
  - `v3_train`: Contains the training data, divided into two subfolders:
    - `images`: Contains the satellite images used for training.
    - `masks`: Contains the corresponding masks for the training images.
  - `v3_val`: Contains the validation data, also divided into two subfolders:
    - `images`: Contains the satellite images used for validation.
    - `masks`: Contains the corresponding masks for the validation images.
  - `v2.zip`: contains the v2 dataset, similar structure as v3
  
  
- **Files:**
  - `fix_tif_masks.ipynb`: Used to transform masks from ArcGIS into a usable format. ArcGIS outputs masks where 0 represents the background and 2 represents roads. This script switches the values to 0 (background) and 1 (road).
  - `resize_tifs.ipynb`: Used to resize .tif files into different sizes. This script is optional and is not needed if the file sizes fetched from ArcGIS are already appropriate.
  - `create_dataset_from_tif.ipynb`: Used to create a dataset (matching PNG images and masks) from .tif files. 

### Training
This folder contains scripts and resources for training the models.

- **Folders:**
  - `scripts`: Contains training scripts for Compute Canada, created using `create_scripts.ipynb`.

- **Files:**
  - `model_pytorch_yoda.py`: Used to train models with PyTorch on the Yoda cluster. The scripts are **not** to be used with this code.
  - `model_tf.py`: Used to train models on Compute Canada using the generated scripts.
  - `run_all.sh`: A shell script that runs all training scripts in the current directory.

### Results
This folder contains the results from training, including model predictions, evaluations, and trained model weights.
For more information, refer to [here](Results/Readme.md)

- **Folders:**
  - `v3_weights.zip`: Contains the trained model weights for various crop sizes and architectures.
  - `v2_results.zip`: Contains the v2 dataset results and weights.
  - `v3_val_small`: Contains all v3 related images, such as original tifs and results.

- **Codes:**
  - `display.ipynb`: Used to display an image, its corresponding mask, and a model prediction for visualization purposes.
  - `find_best_val_f1_score.sh`: A shell script used to find the best model based on the F1-score from the `slurm-XXXXXX.out` output files.
  - `make_figure.ipynb`: Creates a figure comparing the results of all models (2x3 images) for evaluation. The output figure is saved in the `word` folder.
  - `make_masks_visibles.ipynb`: Converts the predicted masks to a visible format (from 0 and 1 to 0 and 255).
  - `make_prediction.ipynb`: Generates predictions for a single model.
  - `multiple_preds.ipynb`: Generates predictions for all models based on the weights stored in the `weights` folder.
  - `reoirt_roads`: a small report regardings the results, both in word and pdf format.
