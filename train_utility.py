import os
import sys
import csv
import logging
import shutil
import tempfile
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import nibabel as nib
import monai
import os
import json
from datetime import datetime
from monai.apps import download_and_extract
from monai.config import print_config
from monai.data import DataLoader, ImageDataset
from sklearn.metrics import roc_auc_score
from fileutility import (
    generate_file_and_label_lists,
    generate_file_and_label_lists_from_extracted_images,
)

def train_model(model, train_loader, optimizer, loss_function, device, max_epochs, tb_writer, train_ds, epoch):
    model.train()
    y_true_train, y_pred_train = [], []
    epoch_loss, step = 0, 0
    train_accuracy_values = []  # Initialize list to store training accuracy

    print(f"\n{'-' * 10}\nepoch {epoch + 1}/{max_epochs}")

    for batch_data in train_loader:
        step += 1
        inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
        inputs = torch.squeeze(torch.transpose(inputs, 1, 4), -1)  # Swap channel and z-axis and remove the last dimension if it's of size 1
        
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
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

    return epoch_loss, auc_train, average_train_accuracy

def eval_model(model, val_loader, loss_function, device):
    model.eval()
    y_true, y_pred = [], []
    num_correct, metric_count, val_loss = 0.0, 0, 0  # Initialize validation loss

    with torch.no_grad():
        for val_data in val_loader:
            val_images, val_labels = val_data[0].to(device), val_data[1].to(device)
            val_images = torch.squeeze(torch.transpose(val_images, 1, 4), -1)  # Swap channel and z-axis and remove the last dimension if it's of size 1

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
    y_true = np.argmax(y_true, axis=1)
    y_pred = np.array(y_pred).reshape(-1, 1)
    auc = roc_auc_score(y_true, y_pred)
    print(f"AUC: {auc}")

    return val_loss, auc, metric

def load_data(file_path, csv_file, train_transforms, val_transforms, pin_memory):
    # Generate file and label lists
    file_list, label_list = generate_file_and_label_lists_from_extracted_images(file_path, csv_file)
    # Map labels 1 and 2 to class 0, and label 3 to class 1
    label_list = [0 if int(label) in [1, 2] else 1 for label in label_list]
    # Convert labels to one-hot format for binary classifier training
    labels = torch.nn.functional.one_hot(torch.as_tensor(label_list)).float()

    # Define nifti dataset and data loader
    train_ds = ImageDataset(image_files=file_list[:250], labels=labels[:250], transform=train_transforms)
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=2, pin_memory=pin_memory)
    val_ds = ImageDataset(image_files=file_list[-50:], labels=labels[-50:], transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=2, pin_memory=pin_memory)

    return train_loader, val_loader

def write_training_info_to_json(model, optimizer, config, loss_function, weights, base_write_path):
    training_info = {
        "model_name": type(model).__name__,
        "pretrained": True,  # or False, depending on your case
        "optimizer": type(optimizer).__name__,
        "learning_rate": config.learning_rate,
        "dropout": model.dropout,  # assuming your model has a 'dropout' attribute
        "loss_function": type(loss_function).__name__,
        "class_weights": weights,
        # add any other parameters you want to track
    }

    # Write the training parameters to a JSON file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(base_write_path, f"training_info_{timestamp}.json")
    with open(file_path, "w") as file:
        json.dump(training_info, file, indent=4)