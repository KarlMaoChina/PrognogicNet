import os
import sys
import csv
import logging
import shutil
import tempfile
import matplotlib.pyplot as plt
import numpy as np
import torch
import nibabel as nib
import monai
import random
import json
from datetime import datetime
from monai.apps import download_and_extract
from monai.config import print_config
from monai.data import DataLoader, ImageDataset
from monai.networks.nets import DenseNet121
from monai.transforms import (
    Compose,
    EnsureChannelFirst,
    RandAffine,
    RandFlip,
    RandGaussianNoise,
    RandRotate90,
    RandZoom,
    Resize,
    ScaleIntensity,
    NormalizeIntensity,
    Rand3DElastic,
    RandCropByPosNegLabel,
    Lambda,
)
from monai.utils.misc import first
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score
from fileutility import (
    generate_file_and_label_lists,
    generate_file_and_label_lists_from_extracted_images,
)
from modelresnet import CustomResNet
from custom_resnet_full import CustomEfficientNet
from train_utility import train_model,eval_model,load_data,write_training_info_to_json
from train_config import TrainConfig

def main():
    config = TrainConfig()
    # Open the CSV file in write mode and create a CSV writer object

    with open(config.csv_file, 'w', newline='') as file:
        csv_writer = csv.writer(file)
        # Write the header row to the CSV file
        csv_writer.writerow(["Epoch", "Train Loss", "Train AUC", "Train Accuracy", "Validation Loss", "Validation AUC", "Accuracy"])
    
    # Check CUDA availability and set device
    pin_memory = config.pin_memory
    device = config.device

    # Set up logging
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    print_config()

    # Define file paths and column numbers for data extraction
    file_path = config.file_path
    csv_file_labels = config.csv_file_labels
    save_path = config.save_path
    column_a = config.column_a
    column_m = config.column_m

    # Generate file and label lists
    file_list, label_list = generate_file_and_label_lists_from_extracted_images(file_path, csv_file_labels)
    print(file_list)
    print(label_list)
    # Map labels 1 and 2 to class 0, and label 3 to class 1
    label_list = [0 if int(label) in [1, 2] else 1 for label in label_list]

    # Convert labels to one-hot format for binary classifier training
    labels = torch.nn.functional.one_hot(torch.as_tensor(label_list)).float()
    print(labels)

    train_transforms = Compose([
        ScaleIntensity(), 
        EnsureChannelFirst(),
        RandRotate90(),
        RandAffine(),
        RandFlip(),
        RandZoom(min_zoom=0.7, max_zoom=1.3, prob=0.5),
        RandGaussianNoise(),
        NormalizeIntensity(),
    ])

    val_transforms = Compose([
        ScaleIntensity(), 
        EnsureChannelFirst(),    
        NormalizeIntensity(),
    ])

    # Define nifti dataset and data loader
    check_ds = ImageDataset(image_files=file_list, labels=labels, transform=train_transforms)
    check_loader = DataLoader(check_ds, batch_size=config.batch_size, num_workers=config.num_workers, pin_memory=pin_memory)

    # Check the first item in the loader
    im, label = first(check_loader)
    im = torch.transpose(im,1,4)
    im = torch.squeeze(im, -1)  # Remove the last dimension if it's of size 1
    print(type(im), im.shape, label, label.shape)

    # Create training and validation data loaders
    train_ds = ImageDataset(image_files=file_list[:250], labels=labels[:250], transform=train_transforms)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=pin_memory)
    val_ds = ImageDataset(image_files=file_list[-50:], labels=labels[-50:], transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=config.num_workers, pin_memory=pin_memory)

    # Define the model
    pretrained_weights_path = "/home/maoshufan/JIA2023819/pretrain_weight/efficientnet-b0-355c32eb.pth"
    model = CustomEfficientNet(weights_path=pretrained_weights_path, pretrained=True,freeze=True).to(device)

    input_data = torch.rand((1, 3, 128, 128)).to(device)
    ouput = model(input_data)
    print(ouput)
    
    # Define the loss function with class weights
    weights = config.weights
    class_weights = torch.FloatTensor(weights).cuda()
    loss_function = torch.nn.CrossEntropyLoss(weight=class_weights)

    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), config.learning_rate)

    # Start a typical PyTorch training
    val_interval = config.val_interval
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = []
    metric_values = []
    tb_writer = SummaryWriter()
    max_epochs = config.max_epochs

    # Write training information to a JSON file
    write_training_info_to_json(model, optimizer, config, loss_function, weights,base_write_path='/home/maoshufan/JIA2023819/train_info')

    for epoch in range(max_epochs):
        epoch_loss, auc_train, average_train_accuracy = train_model(model, train_loader, optimizer, loss_function, device, max_epochs, tb_writer, train_ds,epoch)
        if (epoch + 1) % val_interval == 0:
            val_loss, auc, metric = eval_model(model, val_loader, loss_function, device)
        else:
            # If it's not a validation epoch, set validation loss and AUC to None
            val_loss = 'N/A'
            auc = 'N/A'
            metric = 0
            
        with open(config.csv_file, 'a', newline='') as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow([epoch + 1, epoch_loss, auc_train, average_train_accuracy, val_loss, auc, metric])

            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), "best_metric_model_classification2d_array.pth")
                print("saved new best metric model")

            print(f"Current epoch: {epoch+1} current accuracy: {metric:.4f} ")
            print(f"Best accuracy: {best_metric:.4f} at epoch {best_metric_epoch}")
            tb_writer.add_scalar("val_accuracy", metric, epoch + 1)
            
    print(f"Training completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    tb_writer.close()

if __name__ == "__main__":
    main()