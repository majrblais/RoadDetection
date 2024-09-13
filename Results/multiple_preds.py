#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
os.environ["SM_FRAMEWORK"] = "tf.keras"
import numpy as np
import segmentation_models as sm
import cv2
from tensorflow import keras
from multiprocessing import Pool
# Define the models with their weights paths and corresponding crop sizes
#    'densenet201_1e-05_Unet_1024_noaug': {'weights_path': './tosave/densenet201_1e-05_Unet_1024_noaug/Unet_densenet201_1e-05.h5','crop_size': 1024},


'''
    'densenet121_1e-05_Unet_1024_aug': {
        'weights_path': './tosave/densenet121_1e-05_Unet_1024_aug/Unet_densenet121_best.h5',
        'crop_size': 1024
    },
    

'''

# Define the models with their weights paths and corresponding crop sizes
models_info = {
    'densenet121_1e-05_Unet_1024_noaug': {
        'weights_path': './tosave/densenet121_1e-05_Unet_1024_noaug/Unet_densenet121_best.h5',
        'crop_size': 1024
    },
    'densenet121_1e-05_Unet_128_aug': {
        'weights_path': './tosave/densenet121_1e-05_Unet_128_aug/Unet_densenet121_best.h5',
        'crop_size': 128
    },
    'densenet121_1e-05_Unet_128_noaug': {
        'weights_path': './tosave/densenet121_1e-05_Unet_128_noaug/Unet_densenet121_best.h5',
        'crop_size': 128
    },
    'densenet121_1e-05_Unet_256_aug': {
        'weights_path': './tosave/densenet121_1e-05_Unet_256_aug/Unet_densenet121_best.h5',
        'crop_size': 256
    },
    'densenet121_1e-05_Unet_256_noaug': {
        'weights_path': './tosave/densenet121_1e-05_Unet_256_noaug/Unet_densenet121_best.h5',
        'crop_size': 256
    },
    'densenet121_1e-05_Unet_512_aug': {
        'weights_path': './tosave/densenet121_1e-05_Unet_512_aug/Unet_densenet121_best.h5',
        'crop_size': 512
    },
    'densenet121_1e-05_Unet_512_noaug': {
        'weights_path': './tosave/densenet121_1e-05_Unet_512_noaug/Unet_densenet121_best.h5',
        'crop_size': 512
    }
}

# Define input/output directories
input_images_dir = './v3_val_small/images'
output_dir = './mv_predictions'
os.makedirs(output_dir, exist_ok=True)

# Function to crop images with overlap if not multiple of crop size
def crop_and_resize_image(image, crop_size):
    height, width = image.shape[:2]
    crops = []
    for y in range(0, height, crop_size):
        for x in range(0, width, crop_size):
            x_end = min(x + crop_size, width)
            y_end = min(y + crop_size, height)
            crop = image[y:y_end, x:x_end]
            # Only resize if necessary
            if crop.shape[0] != crop_size or crop.shape[1] != crop_size:
                crop = cv2.resize(crop, (crop_size, crop_size))
            crops.append((crop, x, y))
    return crops

# Function to stitch predictions together into one image
def stitch_predictions(predictions, full_image_size):
    full_height, full_width = full_image_size
    stitched_image = np.zeros((full_height, full_width), dtype=np.uint8)
    
    for pred, x, y in predictions:
        h, w = pred.shape

        # Ensure pred fits into the image by adjusting the size if it exceeds boundaries
        h = min(h, full_height - y)
        w = min(w, full_width - x)

        stitched_image[y:y+h, x:x+w] = np.maximum(stitched_image[y:y+h, x:x+w], pred[:h, :w])
    
    return stitched_image

# Predict crops in a batch for efficiency
def predict_crops_in_batch(crops, model, crop_size):
    crop_batch = []
    positions = []
    for crop, x, y in crops:
        resized_crop = cv2.resize(crop, (crop_size, crop_size))
        crop_batch.append(resized_crop)
        positions.append((x, y))
    
    crop_batch = np.array(crop_batch)
    predictions = model.predict(crop_batch, batch_size=len(crop_batch))
    
    processed_predictions = []
    for i, (x, y) in enumerate(positions):
        pred = (predictions[i] > 0.5).astype(np.uint8) * 255
        processed_predictions.append((pred[:, :, 0], x, y))
        
    return processed_predictions

# Post-processing methods
def apply_morphological_operations(image, kernel_size=5):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

def apply_contour_filtering(image, area_threshold=500):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_image = image.copy()
    for contour in contours:
        if cv2.contourArea(contour) < area_threshold:
            cv2.drawContours(filtered_image, [contour], -1, 0, -1)  # Remove small contours
    return filtered_image

def apply_connected_components_filtering(image, size_threshold=500):
    num_labels, labels_im = cv2.connectedComponents(image)
    filtered_image = image.copy()
    for label in range(1, num_labels):  # Start from 1 to skip the background
        mask = labels_im == label
        if np.sum(mask) < size_threshold:
            filtered_image[mask] = 0
    return filtered_image

# Main processing loop
for model_name, model_info in models_info.items():
    print(f"Processing with model: {model_name}")
    
    # Load the model once
    BACKBONE = model_name.split('_')[0]
    preprocess_input = sm.get_preprocessing(BACKBONE)
    model = sm.Unet(BACKBONE, classes=1, activation='sigmoid')
    model.load_weights(model_info['weights_path'])

    crop_size = model_info['crop_size']

    # Loop through images in the input folder
    for image_file in os.listdir(input_images_dir):
        image_path = os.path.join(input_images_dir, image_file)
        image = cv2.imread(image_path)
        original_size = image.shape[:2]
        print(f"Processing image: {image_file}, size: {original_size}, crop size: {crop_size}")

        # Preprocess and crop the image
        preprocessed_image = preprocess_input(image)
        crops = crop_and_resize_image(preprocessed_image, crop_size)

        # Batch predict crops
        predictions = predict_crops_in_batch(crops, model, crop_size)

        # Stitch predictions together
        stitched_image = stitch_predictions(predictions, original_size)

        # Save original stitched image
        original_output_path = os.path.join(output_dir, f"{model_name}_{image_file}_stitched.png")
        cv2.imwrite(original_output_path, stitched_image)
        print(f"Saved original stitched image at {original_output_path}")

        # Apply morphological operations
        morph_output = apply_morphological_operations(stitched_image)
        morph_output_path = os.path.join(output_dir, f"{model_name}_{image_file}_morph.png")
        cv2.imwrite(morph_output_path, morph_output)
        print(f"Saved morphological operations output at {morph_output_path}")

        # Apply contour filtering
        contour_output = apply_contour_filtering(stitched_image)
        contour_output_path = os.path.join(output_dir, f"{model_name}_{image_file}_contour.png")
        cv2.imwrite(contour_output_path, contour_output)
        print(f"Saved contour filtering output at {contour_output_path}")

        # Apply connected components filtering
        cc_output = apply_connected_components_filtering(stitched_image)
        cc_output_path = os.path.join(output_dir, f"{model_name}_{image_file}_cc.png")
        cv2.imwrite(cc_output_path, cc_output)
        print(f"Saved connected components filtering output at {cc_output_path}")

print("All predictions and post-processed outputs saved.")


# In[ ]:




