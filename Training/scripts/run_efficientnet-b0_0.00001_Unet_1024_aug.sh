#!/bin/bash
#SBATCH --account=rrg-akhloufi
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=0-23:10:00

cd /home/emb9357/
module load cuda
module load opencv
source tensorflow/bin/activate
cd scratch/roads

cd $SLURM_TMPDIR
cp /home/emb9357/scratch/roads/1024_aug.tar . # Copying to local storage
tar -xvf 1024_aug.tar

mv 1024_aug/train ./
mv 1024_aug/val ./

cp /home/emb9357/scratch/roads/model_tf.py .

# Pass arguments to the script
nohup python -u model_tf.py --encoder efficientnet-b0 --lr 0.00001 --model Unet --size 1024 --aug aug
