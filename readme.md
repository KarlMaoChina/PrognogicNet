# `Project-ProgNos`

This script is used for training a 2D classification model with data augmentation. It uses the MONAI framework and PyTorch for the implementation. This is a project for jiaxing no 1 hospital. This is still a early day work.

## Dependencies

- `os`
- `sys`
- `csv`
- `logging`
- `shutil`
- `tempfile`
- `matplotlib`
- `numpy`
- `torch`
- `nibabel`
- `monai`
- `random`
- `json`
- `datetime`
- `sklearn`
- `fileutility`
- `modelresnet`
- `custom_resnet_full`
- `train_utility`
- `train_config`

## Main Functionality

The script starts by setting up the configuration, logging, and defining file paths for data extraction. It then generates file and label lists from the extracted images. The labels are mapped to binary classes and converted to one-hot format for binary classifier training.

Data augmentation is applied to the training data using MONAI's `Compose` function. The transformations include scaling intensity, ensuring the first channel, random rotation, affine transformation, flipping, zooming, and adding Gaussian noise.

The script then defines the Nifti dataset and data loader, and checks the first item in the loader. It creates training and validation data loaders, and defines the model (`CustomEfficientNet` in this case).

The model is trained using the `Adam` optimizer and `CrossEntropyLoss` as the loss function. The training process includes validation at certain intervals, and the best model is saved based on the validation accuracy.

The training information, including the epoch, loss, AUC, and accuracy for both training and validation, is written to a CSV file.

## Usage

To run the script, use the following command:
python train_2d_xy_seg_aug_simp.py


## Configuration

The script uses a `TrainConfig` class to manage all the training configurations. This includes the CSV file path, device settings, file paths for data extraction, column numbers for data extraction, batch size, number of workers, pin memory, class weights, learning rate, validation interval, and maximum epochs.

## Data Preparation

The script generates file and label lists from the extracted images. The labels are mapped to binary classes (labels 1 and 2 to class 0, and label 3 to class 1) and converted to one-hot format for binary classifier training.

## Data Augmentation

Data augmentation is applied to the training data using MONAI's `Compose` function. The transformations include scaling intensity, ensuring the first channel, random rotation, affine transformation, flipping, zooming, and adding Gaussian noise. The validation data undergoes a simpler transformation process, including scaling intensity, ensuring the first channel, and normalizing intensity.