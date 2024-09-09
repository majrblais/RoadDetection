#!/bin/bash

# Navigate to the directory containing your scripts
cd scripts

# Loop over all .sh files in the directory and submit each one using sbatch
for script in *.sh; do
    echo "Submitting $script"
    sbatch "$script"
done