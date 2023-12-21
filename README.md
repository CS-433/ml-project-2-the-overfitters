# Measuring road network expansion and its effect

## Files

### run.ipynb
This is the main notebook used to load the model and dataset, and train the model.

### src/model.py
Contains the U-Net model defined in the **UNet** python class.

### src/dataset.py
Contains the dataset creation and processing part defined in the **SatelliteRoadDataset** class.

### src/runners.py
Contains the train and validate functions.

### src/utils.py
Contains the utility function to reproject the Landsat images and the function to calculate the class weight for the binary cross entropy loss function.

## Files related to task 1
The folder `task-1` contains files related to the analysis of the OSM and GRoad data and that are not directly related to our machine learning model.