import os
os.environ["SM_FRAMEWORK"] = "tf.keras"

import glob
import numpy as np
import segmentation_models as sm

from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import argparse

# Argument parser for command-line arguments
parser = argparse.ArgumentParser(description='Segmentation model training script')
parser.add_argument('--encoder', type=str, default='efficientnetb7', help='Name of the encoder')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
parser.add_argument('--model', type=str, default='Unet', help='Model type from segmentation_models')
parser.add_argument('--size', type=str, default='128', help='size of image')
parser.add_argument('--aug', type=str, default='aug', help='data augmentation')

args = parser.parse_args()

# Create a unique directory for this run based on the arguments
run_dir = f"/home/emb9357/scratch/roads/runs/{args.encoder}_{args.lr}_{args.model}_{args.size}_{args.aug}"
os.makedirs(run_dir, exist_ok=True)

# Select backbone and LR
BACKBONE = args.encoder
preprocess_input = sm.get_preprocessing(BACKBONE)
CLASSES = ['road']
LR = args.lr

# Set training images and mask path
x_train_dir = "./train/i/"
y_train_dir = "./train/m/"

# Set Validation images and mask path
x_valid_dir = "./val/i/"
y_valid_dir = "./val/m/"

# Create two generators, first one for the images, second for the masks
data_generator2 = ImageDataGenerator(preprocessing_function=preprocess_input)
data_generator = ImageDataGenerator()

# Flow from directory
x_generator = data_generator2.flow_from_directory(directory=x_train_dir, target_size=(args.size,args.size), batch_size=2, seed=42, class_mode=None, classes=None)
y_generator = data_generator.flow_from_directory(directory=y_train_dir, target_size=(args.size,args.size), batch_size=2, seed=42, class_mode=None, classes=None)

valx_generator = data_generator2.flow_from_directory(directory=x_valid_dir, target_size=(args.size,args.size), batch_size=2, seed=42, class_mode=None, classes=None)
valy_generator = data_generator.flow_from_directory(directory=y_valid_dir, target_size=(args.size,args.size), batch_size=2, seed=42, class_mode=None, classes=None) 

# Combine image and mask directory to yield images and masks together
def combine_generator(gen1, gen2):
    while True:
        yield(next(gen1), next(gen2))

# Training and validation combination
generator = combine_generator(x_generator, y_generator)
val_generator = combine_generator(valx_generator, valy_generator)

# Model creation
model = getattr(sm, args.model)(BACKBONE, classes=1, activation='sigmoid', encoder_weights=None)
optim = keras.optimizers.Adam(LR)
dice_loss = sm.losses.DiceLoss()
focal_loss = sm.losses.BinaryFocalLoss()
total_loss = dice_loss + (1 * focal_loss)
metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]
model.compile(optim, total_loss, metrics)

# Fit the model
model.fit(generator, steps_per_epoch=2000, epochs=200, validation_data=val_generator, validation_steps=55)

# Save the model
model.save(os.path.join(run_dir, f'{args.model}_{BACKBONE}_{str(LR)}.h5'))
