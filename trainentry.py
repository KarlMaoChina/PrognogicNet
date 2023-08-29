import logging
import os
import sys
import shutil
import tempfile
from fileutility import generate_file_and_label_lists
from monai.metrics import compute_confusion_matrix_metric, get_confusion_matrix

import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np

def main():
    import monai
    from monai.apps import download_and_extract
    from monai.config import print_config
    from monai.data import DataLoader, ImageDataset
    from monai.transforms import (
        EnsureChannelFirst,
        Compose,
        RandRotate90,
        Resize,
        ScaleIntensity,
    )

    pin_memory = torch.cuda.is_available()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    print_config()
    file_path = "/data/maoshufan/jiadata/T2WI-1/images"
    excel_path = "/data/maoshufan/jiadata/T2WI-1/result.csv"
    column_a = 1
    column_m = 13

    file_list, label_list = generate_file_and_label_lists(file_path, excel_path)
    print(file_list)
    print(label_list)
    label_list = [int(label) - 1 for label in label_list]

    # Represent labels in one-hot format for binary classifier training,
    # BCEWithLogitsLoss requires target to have same shape as input
    labels = torch.nn.functional.one_hot(torch.as_tensor(label_list)).float()
    print(labels)
    # Define transforms
    train_transforms = Compose([ScaleIntensity(), EnsureChannelFirst(), Resize((256, 256, 32)), RandRotate90()])

    val_transforms = Compose([ScaleIntensity(), EnsureChannelFirst(), Resize((256, 256, 32))])

    # Define nifti dataset, data loader
    check_ds = ImageDataset(image_files=file_list, labels=labels, transform=train_transforms)
    check_loader = DataLoader(check_ds, batch_size=4, num_workers=2, pin_memory=pin_memory)

    im, label = monai.utils.misc.first(check_loader)
    print(type(im), im.shape, label, label.shape)

    # create a training data loader
    train_ds = ImageDataset(image_files=file_list[:70], labels=labels[:70], transform=train_transforms)
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=2, pin_memory=pin_memory)

    # create a validation data loader
    val_ds = ImageDataset(image_files=file_list[-30:], labels=labels[-30:], transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=2, num_workers=2, pin_memory=pin_memory)
    model = monai.networks.nets.DenseNet121(spatial_dims=3, in_channels=1, out_channels=3).to(device)
    input_data = torch.rand((1, 1, 256, 256, 32)).to(device)
    ouput = model(input_data)
    print(ouput)
    
    #loss_function = torch.nn.CrossEntropyLoss()
    weights = [0.1, 0.1, 0.8]  # adjust as needed
    class_weights = torch.FloatTensor(weights).cuda()
    loss_function = torch.nn.CrossEntropyLoss(weight=class_weights)
    # loss_function = torch.nn.BCEWithLogitsLoss()  # also works with this data

    optimizer = torch.optim.Adam(model.parameters(), 1e-4)

    # start a typical PyTorch training
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
                with torch.no_grad():
                    val_outputs = model(val_images)
                    value = torch.eq(val_outputs.argmax(dim=1), val_labels.argmax(dim=1))
                    metric_count += len(value)
                    num_correct += value.sum().item()

            metric = num_correct / metric_count
            metric_values.append(metric)
            y_pred = val_outputs.argmax(dim=1).detach().cpu().numpy()
            y_pred = torch.from_numpy(y_pred).float()
            y = val_labels.argmax(dim=1).detach().cpu().numpy()
            y = torch.from_numpy(y).float()
            confusion_matrix = get_confusion_matrix(y, y_pred)
            sensitivity = compute_confusion_matrix_metric(sensitivity, confusion_matrix)
            print(f"Sensitivity: {sensitivity}")
            specificity = compute_confusion_matrix_metric(specificity, confusion_matrix)
            print(f"Specificity: {specificity}")

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