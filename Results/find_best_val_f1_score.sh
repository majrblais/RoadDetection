#!/bin/bash

# Initialize an associative array to store the best val_f1-score for each size/augmentation combination
declare -A best_file
declare -A highest_val_f1_score
declare -A best_epoch

# Initialize sizes and augmentation types
sizes=("128" "256" "512" "1024")
augmentations=("aug" "noaug")

# Initialize highest scores and epoch for each size/augmentation combination
for size in "${sizes[@]}"; do
    for aug in "${augmentations[@]}"; do
        highest_val_f1_score["$size_$aug"]=0
        best_file["$size_$aug"]=""
        best_epoch["$size_$aug"]=0
    done
done

# Function to convert scientific notation to a float
convert_to_float() {
  printf "%.10f\n" "$1"
}

# Loop through all .out files in the current directory
for file in *.out; do
	echo $file
  # Extract val_f1-score values from the file along with the epoch
  val_f1_scores_and_epochs=$(grep -P -n 'val_f1-score: [0-9]+\.[0-9]+(e-?[0-9]+)?' "$file")

  # Extract the size and augmentation type from the file content (assuming size_augtype/val/m/data/XX.png)
  size_and_aug=$(grep -oP '[0-9]+_(aug|noaug)' "$file" | head -1)

  # Extract size and augmentation from the string
  size=$(echo "$size_and_aug" | grep -oP '^[0-9]+')
  aug=$(echo "$size_and_aug" | grep -oP '(aug|noaug)')

  # Loop through the extracted val_f1-score values and epochs
  while IFS=: read -r epoch_line content; do
    epoch=$(grep -oP 'Epoch \K[0-9]+' <<< "$content")
    score=$(grep -oP 'val_f1-score: \K[0-9]+\.[0-9]+(e-?[0-9]+)?' <<< "$content")

    # Convert score to float if in scientific notation
    float_score=$(convert_to_float "$score")

    # Compare the current score with the highest score for the given size and augmentation
    if (( $(echo "$float_score > ${highest_val_f1_score["$size_$aug"]}" | bc -l) )); then
      highest_val_f1_score["$size_$aug"]=$float_score
      best_file["$size_$aug"]=$file
      best_epoch["$size_$aug"]=$epoch
    fi
  done <<< "$val_f1_scores_and_epochs"
done


#currettly doesnt work, the path is wrong`best_out_file`
# Create the "tosave" directory if it doesn't exist
tosave_dir="./tosave"
mkdir -p "$tosave_dir"
echo "saving..."
# Output the best file, highest val_f1-score, corresponding training f1-score, and copy relevant run folder
for size in "${sizes[@]}"; do
    for aug in "${augmentations[@]}"; do
        best_out_file=${best_file["$size_$aug"]}
        best_score=${highest_val_f1_score["$size_$aug"]}
        best_epoch_num=${best_epoch["$size_$aug"]}

        if [ -n "$best_out_file" ]; then
            echo "For size: $size, augmentation: $aug"
            echo "File with the highest val_f1-score: $best_out_file"
            echo "Highest val_f1-score: $best_score"
            echo "Epoch with the highest val_f1-score: $best_epoch_num"

            # Extract the corresponding training f1-score for the best epoch
            training_f1_score=$(grep -P "Epoch $best_epoch_num/[0-9]+ .* f1-score: [0-9]+\.[0-9]+" "$best_out_file" | grep -oP 'f1-score: \K[0-9]+\.[0-9]+')
            echo "Training f1-score at epoch $best_epoch_num: $training_f1_score"

            # Use sacct to gather job information for the best .out file (assuming the file name is in the format slurm-<job_id>.out)
            job_id=$(echo "$best_out_file" | grep -oP '(?<=slurm-)[0-9]+')

            echo "SLURM job details for job ID: $job_id"
            sacct -j "$job_id" --format=JobID,JobName,MaxRSS,Elapsed,State

            
            # Copy the run directory to the "tosave" directory
            run_dir="/home/emb9357/scratch/roads/runs/${best_out_file}_${size}_${aug}"
            if [ -d "$run_dir" ]; then
                echo "Copying $run_dir to $tosave_dir"
                cp -r "$run_dir" "$tosave_dir"
            else
                echo "Run directory $run_dir not found."
            fi

            echo "-----------------------------------"
        fi
    done
done
