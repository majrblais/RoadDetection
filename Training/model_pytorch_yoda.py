import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader, Dataset as BaseDataset, Subset
from sklearn.model_selection import train_test_split
import albumentations as albu
from segmentation_models_pytorch import utils

# Directory paths
DATA_DIR = './data/'
x_train_dir = os.path.join(DATA_DIR, 'train/i/')
y_train_dir = os.path.join(DATA_DIR, 'train/m/')
x_valid_dir = os.path.join(DATA_DIR, 'val/i/')
y_valid_dir = os.path.join(DATA_DIR, 'val/m/')

# Dataset class
class Dataset(BaseDataset):
    """Dataset. Read images, apply augmentation and preprocessing transformations."""
    
    def __init__(self, images_dir, masks_dir, classes=None, augmentation=None, preprocessing=None, return_original=False):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
        
        self.class_values = [0, 1]  # Binary segmentation
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.return_original = return_original
    
    def __getitem__(self, i):
        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0)
        original_image = image.copy() 

        # Ensure the mask is binary (0 and 1)
        mask[mask != 0] = 1
        mask = np.expand_dims(mask, axis=-1).astype('float')
        
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
        if self.return_original:
            return image, mask, original_image
        else:
            return image, mask
        
    def __len__(self):
        return len(self.ids)

def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.PadIfNeeded(384, 480)
    ]
    return albu.Compose(test_transform)

def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

def simple_preprocessing(image, **kwargs):
    return image / 255.0

def get_simple_preprocessing():
    """Construct preprocessing transform"""
    _transform = [
        albu.Lambda(image=simple_preprocessing),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)

def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform"""
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)

# Preprocessing function
ENCODERS = ['densenet121', 'vgg19', 'timm-mobilenetv3_large_100', 'efficientnet-b7', 'resnet18', 'mit_b0','timm-gernet_s','xception',]
ENCODER_WEIGHTS = 'imagenet'
ACTIVATION = 'sigmoid'  # could be None for logits or 'softmax2d' for multiclass segmentation
DEVICE = 'cuda'

# Create full training dataset
full_dataset = Dataset(
    x_train_dir, 
    y_train_dir, 
    preprocessing=get_simple_preprocessing(),
)

# Split dataset into training and validation
train_indices, valid_indices = train_test_split(
    np.arange(len(full_dataset)), test_size=0.2, random_state=42
)

train_dataset = Subset(full_dataset, train_indices)
valid_dataset = Subset(full_dataset, valid_indices)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=12)
valid_loader = DataLoader(valid_dataset, batch_size=2, shuffle=False, num_workers=4)

# Dice/F1 score - https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
# IoU/Jaccard score - https://en.wikipedia.org/wiki/Jaccard_index
loss = utils.losses.DiceLoss()
metrics = [
    utils.metrics.IoU(threshold=0.5),
    utils.metrics.Fscore(threshold=0.5),
]

best_score = 0
best_model = None
best_encoder = None

for ENCODER in ENCODERS:
    print(f"Training with encoder: {ENCODER}")
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
    
    # create segmentation model with pretrained encoder
    model = smp.Unet(
        encoder_name=ENCODER, 
        encoder_weights=ENCODER_WEIGHTS, 
        classes=1, 
        activation=ACTIVATION,
    )

    optimizer = torch.optim.Adam([ 
        dict(params=model.parameters(), lr=0.00001),
    ])

    # create epoch runners
    train_epoch = utils.train.TrainEpoch(
        model, 
        loss=loss, 
        metrics=metrics, 
        optimizer=optimizer,
        device=DEVICE,
        verbose=True,
    )

    valid_epoch = utils.train.ValidEpoch(
        model, 
        loss=loss, 
        metrics=metrics, 
        device=DEVICE,
        verbose=True,
    )

    max_score = 0

    for i in range(0, 150):
        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)
        
        # Update the best model if the current epoch's IoU score is higher than the max_score
        if max_score < valid_logs['fscore']:
            max_score = valid_logs['fscore']
            torch.save(model.state_dict(), f'./best_model_temp_{ENCODER}.pth')
            print('Model saved!')
        
        # Change the learning rate at epoch 100
        if i == 100:
            optimizer.param_groups[0]['lr'] = 1e-6
            print('Decrease decoder learning rate to 1e-6!')

    if max_score > best_score:
        best_score = max_score
        best_model = model
        best_encoder = ENCODER
        
        os.rename(f'./best_model_temp_{ENCODER}.pth', f'./best_model_{ENCODER}.pth')
    print(max_score)
    print('____________________________________________________________')
print(f"Best encoder: {best_encoder} with IoU score: {best_score}")

# Load the best model
best_model.load_state_dict(torch.load(f'./best_model_{best_encoder}.pth'))
best_model = best_model.to(DEVICE)
best_model.eval()

# Create validation dataset from the valid subfolder
valid_dataset_for_prediction = Dataset(
    x_valid_dir, 
    y_valid_dir, 
    preprocessing=get_preprocessing(smp.encoders.get_preprocessing_fn(best_encoder, ENCODER_WEIGHTS)),
    return_original=True  # Ensure original images are returned for prediction
)

valid_loader_for_prediction = DataLoader(valid_dataset_for_prediction, batch_size=2, shuffle=False, num_workers=4)

# Predict and save validation images
results_dir = f"./{best_encoder}_results/"
os.makedirs(os.path.join(results_dir, "images"), exist_ok=True)
os.makedirs(os.path.join(results_dir, "masks"), exist_ok=True)
os.makedirs(os.path.join(results_dir, "predictions"), exist_ok=True)

for i, batch in enumerate(valid_loader_for_prediction):
    images, masks, original_images = batch
    images = images.to(DEVICE)
    masks = masks.to(DEVICE)  # Ensure masks are also on the correct device
    
    with torch.no_grad():
        predictions = best_model(images)
        predictions = predictions.cpu().numpy()
    
    images = images.cpu().numpy()
    original_images = original_images.cpu().numpy()
    masks = masks.cpu().numpy()
    
    for j in range(images.shape[0]):
        image = images[j].transpose(1, 2, 0)  # Change the shape to HWC for saving
        original_image = original_images[j] 
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)        
        mask = (masks[j][0] * 255).astype(np.uint8)
        image = (image * 255).astype(np.uint8)
        prediction = predictions[j][0]
        thresholded_prediction = (prediction > 0.5).astype(np.uint8) * 255

        cv2.imwrite(os.path.join(results_dir, "images", f"val_image_{i * images.shape[0] + j}.png"), original_image)
        cv2.imwrite(os.path.join(results_dir, "masks", f"val_mask_{i * images.shape[0] + j}.png"), mask)
        cv2.imwrite(os.path.join(results_dir, "predictions", f"val_pred_{i * images.shape[0] + j}.png"), thresholded_prediction)

print(f"Images, masks, and predictions saved in the '{best_encoder}_results' folder.")
