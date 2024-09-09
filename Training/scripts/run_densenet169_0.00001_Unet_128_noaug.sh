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
cp /home/emb9357/scratch/roads/128_noaug.tar . # Copying to local storage
tar -xvf 128_noaug.tar

mv 128_noaug/train ./
mv 128_noaug/val ./

cp /home/emb9357/scratch/roads/model_tf.py .

# Pass arguments to the script
nohup python -u model_tf.py --encoder densenet169 --lr 0.00001 --model Unet --size 128 --aug noaug
