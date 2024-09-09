# Results

This directory is used to show results and how to make predictions.

- `val_images`: Contains the validation images.
- `val_masks`: Contains the true masks for the validation images.
- `prediction_masks`: Contains the predicted masks from our best performing model, `Unet_inceptionresnetv2_0.0001.h5`.

## Scripts and Usage

- `find_best_val_f1_score.sh`: This script is used to find the best validation F1-score across all possible SLURM output files. The output of this script is a `slurm-xxxxxxxx.out` file. To identify the model associated with the best score, use the command:
  ```
  sacct -j xxxxxxxx --format=JobID,JobName%50,State,Elapsed,ExitCode
  ```
  (In future work, make this process smoother)

- `make_prediction.ipynb`: This Jupyter notebook is used to make predictions given a model and its weights.

- `make_masks_visibles.ipynb`: This Jupyter notebook is used to visualize the predictions.

## Future Work

- Make the process of finding the best validation F1-score smoother.
