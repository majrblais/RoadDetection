# Results

This directory is used to show results and how to make predictions.

- `v2_results`: Contains the validation predictions and overall results for our v2 dataset.

## Scripts and Usage

- `find_best_val_f1_score.sh`: This script is used to find the best validation F1-score across all possible SLURM output files. The output of this script is a `slurm-xxxxxxxx.out` file. To identify the model associated with the best score, use the command:
  ```
  sacct -j xxxxxxxx --format=JobID,JobName%50,State,Elapsed,ExitCode
  ```
  (In future work, make this process smoother)

- `make_prediction.ipynb`: This Jupyter notebook is used to make predictions given a model and its weights.

- `multiple_preds.ipynb/py` takes in all best models for each size (for both aug and noaug) and predicts the validation tif images. Compared to `make_prediction.ipynb`, this code takes original-sized tif images and crops them to create predictions of the entire (5000x5000) image. Fruthermore, this code uses noise-reducing techniques.

- `make_masks_visibles.ipynb`: This Jupyter notebook is used to visualize the predictions.


- `make_figure.ipynb` is used to make a figure of the predictions

- `display.ipynb` is used to show an image, its mask and a prediction, it is used to iterate throught examples.

## Usage
1. Using the weights of the best models selected (either by validation f1-score or manually), launch either `make_prediction.ipynb` or `multiple_preds.ipynb/py`  for predictions by changing the appropriate weights and parameters. The codes are not automatic and must be manually changed depending on the weights. 
2. Using the results from `multiple_preds.ipynb/py` and by making the masks visible using `make_masks_visibles.ipynb`, the images can be displayed using `display.ipynb`, the code must be modified accordingly.
3. `make_figure.ipynb` takes in results from `multiple_preds.ipynb/py` and visible masks to compare the models, similarly the code must be changed manually.

This repo is only used for preliminary testing is not made for production or comprehension, rather it enables to test approaches for road detection and to give a starting point in future works.
We also provide the report (word/pdf) which explains our results and possible future directions, please read the [report](./report_roads.pdf) for more information.

## Future Work
- Make the process of finding the best validation F1-score smoother.
- Clean code, comment and automatize process.