import os
import sys
import logging
import shutil
import tempfile
import nibabel as nib
import matplotlib.pyplot as plt
import torch
import numpy as np
import monai
from monai.apps import download_and_extract
from monai.config import print_config
from monai.data import DataLoader, ImageDataset
from monai.transforms import (
    EnsureChannelFirst,
    Compose,
    RandRotate90,
    RandAffine,
    RandFlip,
    RandZoom,
    RandGaussianNoise,
    Resize,
    ScaleIntensity,
    Lambda,
)
from monai.networks.nets import DenseNet121
from monai.utils.misc import first
from torch.utils.tensorboard import SummaryWriter
from fileutility import generate_file_and_label_lists,generate_file_and_label_lists_from_extracted_images
from modelresnet import CustomResNet

def main():
    # Check CUDA availability and set device
    pin_memory = torch.cuda.is_available()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set up logging
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    print_config()

    # Define file paths and column numbers for data extraction
    file_path = "/data/maoshufan/jiadata/T2WI-1/images"
    csv_file = "/data/maoshufan/jiadata/T2WI-1/result.csv"
    save_path = "/data/maoshufan/jiadata/T2WI-1/imagesave/"
    column_a = 1
    column_m = 13

    # Generate file and label lists
    file_list, label_list = generate_file_and_label_lists_from_extracted_images(file_path, csv_file)
    print(file_list)
    print(label_list)
    # Map labels 1 and 2 to class 0, and label 3 to class 1
    label_list = [0 if int(label) in [1, 2] else 1 for label in label_list]

    # Convert labels to one-hot format for binary classifier training
    labels = torch.nn.functional.one_hot(torch.as_tensor(label_list)).float()
    print(labels)

    train_transforms = Compose([ScaleIntensity(), EnsureChannelFirst(), Resize((512, 512,3)), RandRotate90(), RandAffine])
    val_transforms = Compose([ScaleIntensity(), EnsureChannelFirst(),  Resize((512, 512,3))])

    # Define nifti dataset and data loader
    check_ds = ImageDataset(image_files=file_list, labels=labels, transform=train_transforms)
    check_loader = DataLoader(check_ds, batch_size=4, num_workers=2, pin_memory=pin_memory)

    # Check the first item in the loader
    im, label = first(check_loader)
    im = torch.transpose(im,1,4)
    im = torch.squeeze(im, -1)  # Remove the last dimension if it's of size 1
    print(type(im), im.shape, label, label.shape)

    # Create training and validation data loaders
    train_ds = ImageDataset(image_files=file_list[:70], labels=labels[:70], transform=train_transforms)
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=2, pin_memory=pin_memory)
    val_ds = ImageDataset(image_files=file_list[-30:], labels=labels[-30:], transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=2, pin_memory=pin_memory)

    # Define the model
    model = CustomResNet().to(device)
    input_data = torch.rand((1, 3, 512, 512)).to(device)
    ouput = model(input_data)
    print(ouput)
    
    # Define the loss function with class weights
    weights = [0.77, 0.23]  # adjust as needed
    class_weights = torch.FloatTensor(weights).cuda()
    loss_function = torch.nn.CrossEntropyLoss(weight=class_weights)

    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), 1e-4)

    # Start a typical PyTorch training
    val_interval = 2
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = []
    metric_values = []
    writer = SummaryWriter()
    max_epochs = 30

    for epoch in range(max_epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        model.train()
        epoch_loss = 0
        step = 0

        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
            inputs = torch.transpose(inputs, 1, 4)  # Swap channel and z-axis
            inputs = torch.squeeze(inputs, -1)  # Remove the last dimension if it's of size 1
            
            for i, input_data in enumerate(inputs):
                input_data = input_data.cpu().numpy()  # Convert the tensor to numpy array
                input_data = np.transpose(input_data, (2,1,0))  # Transpose the dimensions
                nifti_img = nib.Nifti1Image(input_data, np.eye(4))  # Create a NIfTI image
                nib.save(nifti_img, os.path.join(save_path, f"input_data_epoch{epoch+1}_batch{step}_index{i}.nii.gz"))  # Save the NIfTI image

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_len = len(train_ds) // train_loader.batch_size
            print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
            writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)

        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % val_interval == 0:
            model.eval()

            num_correct = 0.0
            metric_count = 0
            for val_data in val_loader:
                val_images, val_labels = val_data[0].to(device), val_data[1].to(device)
                val_images = torch.transpose(val_images, 1, 4)  # Swap channel and z-axis
                val_images = torch.squeeze(val_images, -1)  # Remove the last dimension if it's of size 1
                with torch.no_grad():
                    val_outputs = model(val_images)
                    value = torch.eq(val_outputs.argmax(dim=1), val_labels.argmax(dim=1))
                    metric_count += len(value)
                    num_correct += value.sum().item()

            metric = num_correct / metric_count
            metric_values.append(metric)

            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), "best_metric_model_classification3d_array.pth")
                print("saved new best metric model")

            print(f"Current epoch: {epoch+1} current accuracy: {metric:.4f} ")
            print(f"Best accuracy: {best_metric:.4f} at epoch {best_metric_epoch}")
            writer.add_scalar("val_accuracy", metric, epoch + 1)

    print(f"Training completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    writer.close()

if __name__ == "__main__":
    main()