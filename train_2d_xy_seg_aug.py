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

def main():

    # Open the CSV file in write mode and create a CSV writer object
    with open('training_data.csv', 'w', newline='') as file:
        csv_writer = csv.writer(file)
        # Write the header row to the CSV file
        csv_writer.writerow(["Epoch", "Train Loss", "Train AUC", "Train Accuracy", "Validation Loss", "Validation AUC", "Accuracy"])
    
    # Check CUDA availability and set device
    pin_memory = torch.cuda.is_available()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set up logging
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    print_config()

    # Define file paths and column numbers for data extraction
    file_path = "/data/maoshufan/jiadata/T2WI-1/images_cropxy_full/"
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

    train_transforms = Compose([
        ScaleIntensity(), 
        EnsureChannelFirst(),
        RandRotate90(),
        RandAffine(),
        RandFlip(),
        RandZoom(),
        RandGaussianNoise(),
    ])

    val_transforms = Compose([ScaleIntensity(), EnsureChannelFirst(),])

    # Define nifti dataset and data loader
    check_ds = ImageDataset(image_files=file_list, labels=labels, transform=train_transforms)
    check_loader = DataLoader(check_ds, batch_size=4, num_workers=2, pin_memory=pin_memory)

    # Check the first item in the loader
    im, label = first(check_loader)
    im = torch.transpose(im,1,4)
    im = torch.squeeze(im, -1)  # Remove the last dimension if it's of size 1
    print(type(im), im.shape, label, label.shape)

    # Create training and validation data loaders
    train_ds = ImageDataset(image_files=file_list[:250], labels=labels[:250], transform=train_transforms)
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=2, pin_memory=pin_memory)
    val_ds = ImageDataset(image_files=file_list[-50:], labels=labels[-50:], transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=2, pin_memory=pin_memory)

    # Define the model
    model = CustomResNet().to(device)
    input_data = torch.rand((1, 3, 128, 128)).to(device)
    ouput = model(input_data)
    print(ouput)
    
    # Define the loss function with class weights
    weights = [0.7, 0.3]  # adjust as needed
    class_weights = torch.FloatTensor(weights).cuda()
    loss_function = torch.nn.CrossEntropyLoss(weight=class_weights)

    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), 1e-4)

    # Start a typical PyTorch training
    val_interval = 1
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = []
    metric_values = []
    tb_writer = SummaryWriter()
    max_epochs = 30

    for epoch in range(max_epochs):

        y_true_train = []
        y_pred_train = []
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        model.train()
        epoch_loss = 0
        step = 0
        train_accuracy_values = []  # Initialize list to store training accuracy

        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
            inputs = torch.transpose(inputs, 1, 4)  # Swap channel and z-axis
            inputs = torch.squeeze(inputs, -1)  # Remove the last dimension if it's of size 1
            
            optimizer.zero_grad()
            outputs = model(inputs)

            probabilities = torch.softmax(outputs, dim=1)
            y_true_train.extend(labels.cpu().numpy())
            y_pred_train.extend(probabilities[:, 1].detach().cpu().numpy())  # Probability for the positive class

            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_len = len(train_ds) // train_loader.batch_size
            print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
            tb_writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
            # Calculate training accuracy for each batch
            correct_predictions = torch.eq(outputs.argmax(dim=1), labels.argmax(dim=1))
            train_accuracy = correct_predictions.sum().item() / len(correct_predictions)
            train_accuracy_values.append(train_accuracy)  # Append training accuracy to the list
        
        # Calculate average training accuracy for the epoch
        average_train_accuracy = sum(train_accuracy_values) / len(train_accuracy_values)
        print(f"Average training accuracy for epoch {epoch + 1}: {average_train_accuracy:.4f}")
        
        # Convert the list of true labels to a numpy array and take the argmax along axis 1
        y_true_train = np.argmax(y_true_train, axis=1)
        # Convert the list of predicted probabilities to a numpy array and reshape to (-1, 1)
        y_pred_train = np.array(y_pred_train).reshape(-1, 1)
        # Calculate the Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores
        auc_train = roc_auc_score(y_true_train, y_pred_train)
        # Print the AUC for the training data
        print(f"Train AUC: {auc_train}")

        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % val_interval == 0:
            model.eval()
            y_true = []
            y_pred = []
            num_correct = 0.0
            metric_count = 0
            val_loss = 0  # Initialize validation loss
            for val_data in val_loader:
                val_images, val_labels = val_data[0].to(device), val_data[1].to(device)
                val_images = torch.transpose(val_images, 1, 4)  # Swap channel and z-axis
                val_images = torch.squeeze(val_images, -1)  # Remove the last dimension if it's of size 1
                with torch.no_grad():
                    val_outputs = model(val_images)

                    # Calculate validation loss
                    loss = loss_function(val_outputs, val_labels)
                    val_loss += loss.item()
                    
                    probabilities = torch.softmax(val_outputs, dim=1)
                    y_true.extend(val_labels.cpu().numpy())  # Apply np.argmax here
                    y_pred.extend(probabilities[:, 1].cpu().numpy())  # Probability for the positive class
                    
                    value = torch.eq(val_outputs.argmax(dim=1), val_labels.argmax(dim=1))
                    metric_count += len(value)
                    num_correct += value.sum().item()

            # Calculate average validation loss
            val_loss /= len(val_loader)
            print(f"Validation loss: {val_loss:.4f}")

            metric = num_correct / metric_count
            metric_values.append(metric)
            y_true = np.argmax(y_true, axis=1)
            y_pred = np.array(y_pred).reshape(-1, 1)
            auc = roc_auc_score(y_true, y_pred)
            print(f"AUC: {auc}")

        else:
            # If it's not a validation epoch, set validation loss and AUC to None
            val_loss = 'N/A'
            auc = 'N/A'
            metric = 0
            
        with open('training_data.csv', 'a', newline='') as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow([epoch + 1, epoch_loss, auc_train, average_train_accuracy, val_loss, auc, metric])

            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), "best_metric_model_classification3d_array.pth")
                print("saved new best metric model")

            print(f"Current epoch: {epoch+1} current accuracy: {metric:.4f} ")
            print(f"Best accuracy: {best_metric:.4f} at epoch {best_metric_epoch}")
            tb_writer.add_scalar("val_accuracy", metric, epoch + 1)
            
    print(f"Training completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    tb_writer.close()

if __name__ == "__main__":
    main()