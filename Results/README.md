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

- `make_masks_visibles.ipynb`: This Jupyter notebook is used to visualize the predictions.

- `multiple_preds.ipynb` takes in all best models for each size (for both aug and noaug) and predicts the validation tif images. Compared to `make_prediction.ipynb`, this code takes original-sized tif images and crops them to create predictions of the entire (5000*5000) image.

- `make_figure.ipynb` is used to make a figure of the predictions

- `display.ipynb` is used to show an image, its mask and a prediction, it is used to iterate throught examples.

## Future Work

- Make the process of finding the best validation F1-score smoother.
