# Data Creation
This folder is used to fetch and create our datasets.
First we need to fetch our data using ArcGis then modify that data to be used for deep learning.

## ArcGis Data Fetching
This guide outlines the steps to prepare training data for deep learning in ArcGIS Pro using a specific workflow. We will transform a line shapefile into a raster, resample it, and export the training data using a combination of hillshade and the rasterized lines. 

### Prerequisites

- ArcGIS Pro
- Access to the `Réseau_routier_MAJ_22` shapefile from the `ccnb-innov` organization.
- Access to the hillshade file (e.g., `HS_315`).

### Steps

#### 1. Load the Line Shapefile

Load the `Réseau_routier_MAJ_22` shapefile into ArcGIS Pro.

#### 2. Convert Line Shapefile to Raster

Use the `Feature to Raster` tool to convert the line shapefile to a raster format.

1. **Open the Feature to Raster Tool**:
   - In the Geoprocessing pane, search for and open the `Feature to Raster` tool.

2. **Configure the Tool Parameters**:
   - **Input Features**: `Réseau_routier_MAJ_22`
   - **Field**: Create and use a constant value field (e.g., `ConstantValue`) if not already present.
   - **Output Raster**: Choose a location to save the output raster file (e.g., `Reseau_Routier_Raster`).
   - **Cell Size**: Set the cell size to 3.

   Example:
   ```plaintext
   Feature To Raster
   (Input Features: "Réseau_routier_MAJ_22", Field: "ConstantValue", Output Raster: "Reseau_Routier_Raster", Cell Size: 3)```

#### 3. Load the Hillshade File

Load the hillshade file (`HS_315`) from the GeoNB server or any other appropriate raster file such as world imagery.

#### 4. Export Training Data for Deep Learning

Use the `Export Training Data For Deep Learning` tool to export the data.

1. **Open the Export Training Data For Deep Learning Tool**:
   - In the Geoprocessing pane, search for and open the `Export Training Data For Deep Learning` tool.

2. **Configure the Tool Parameters**:
   - **Input Raster**: The hillshade file (`HS_315`).
   - **Input Feature Class**: `Réseau_routier_MAJ_22`
   - **Additional Input Raster**: The raster file created from the lines shapefile (`Reseau_Routier_Raster`).
   - **Output Folder**: Choose a location to save the output (e.g., `Training_Data_Output`).
   - **Image Format**: `TIFF`
   - **Tile Size X**: `1024`
   - **Tile Size Y**: `1024`
   - **Stride X**: `512`
   - **Stride Y**: `512`
   - **Meta Data Format**: `RCNN_Masks`

   Example:
   ```plaintext
   Export Training Data For Deep Learning
   (Input Raster: "HS_315",
    Input Feature Class: "Réseau_routier_MAJ_22",
    Additional Input Raster: "Reseau_Routier_Raster",
    Output Folder: "Training_Data_Output",
    Image Format: "TIFF",
    Tile Size X: 1024,
    Tile Size Y: 1024,
    Stride X: 512,
    Stride Y: 512,
    Meta Data Format: "RCNN_Masks")```

#### 5. Verify the Output

The tool will create three folders: `images`, `images2`, and `labels`.

- **images**: Contains the exported image tiles.
- **images2**: Contains the corresponding masks created from the additional input raster.
- **labels**: This folder will be empty due to the workaround used.

#### 6. Use the Exported Data

For training purposes, use the images from the `images` folder and the masks from the `images2` folder. The `labels` folder can be ignored.

#### Future Work

- Investigate and find a fix for the empty `labels` folder issue to streamline the process.


## Create Deep Learning Dataset
This directory is used to create data from the images and images2 dataset created from the previous step.



`fix_tif_masks.ipynb` is used to trasnform the masks from ArcGis into a usable format. Currently the output uses 0 for background and 2 for road, this code switches it to 0 and 1.

`resize_tifs.ipynb` is used to resize tifs into different sizes and into png format. Not used depending on the fiel size fetched from Arcgis.

`create_dataset_from_tif.ipynb` is used to create a dataset (matching png images and masks) from tif files. Currently, it creates the training and validation set for sizes of 128,256,512 and 1,024 for both augmentation and no augmentation.

folder `v2` contains our second version of our training data while folders `v3_train` and `v3_val` contain our latest training and validation data.

The data from `create_dataset_from_tif.ipynb` is excluded from the repo due to the size of the created dataset.



The output of `create_dataset_from_tif.ipynb` should be allow users to train segmentation models using deep learning.



