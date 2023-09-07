import torch
import numpy as np
from monai.data import DataLoader, ImageDataset
from monai.transforms import (
    Compose,
    EnsureChannelFirst,
    ScaleIntensity,
    NormalizeIntensity,
)
from custom_resnet_full import CustomEfficientNet
from create_train_val_set import load_from_file
from sklearn.metrics import roc_auc_score


def eval_model(model, val_loader, loss_function, device, threshold=0.5):
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

            y_pred_binary = (probabilities[:, 1] > threshold).cpu().numpy()  # Classify based on the threshold
            value = torch.eq(torch.tensor(y_pred_binary).to(device), val_labels.argmax(dim=1))
            metric_count += len(value)
            num_correct += value.sum().item()

    # Calculate average validation loss
    val_loss /= len(val_loader)
    #print(f"Validation loss: {val_loss:.4f}")

    metric = num_correct / metric_count
    y_true = np.argmax(y_true, axis=1)
    y_pred = np.array(y_pred).reshape(-1, 1)
    auc = roc_auc_score(y_true, y_pred)
    #print(f"AUC: {auc}")

    return val_loss, auc, metric

def main():
    # Define the validation transforms
    val_transforms = Compose([
        ScaleIntensity(),
        EnsureChannelFirst(),    
        NormalizeIntensity(),
    ])

    # Load the validation data
    val_files, val_labels = load_from_file('val_set_T2W1.pkl')  # Replace with your validation data file path
    val_labels = torch.nn.functional.one_hot(torch.as_tensor(val_labels)).float()

    # Define the validation dataset and data loader
    val_ds = ImageDataset(image_files=val_files, labels=val_labels, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=32, num_workers=4, pin_memory=True)  # Adjust batch_size and num_workers as needed

    # Load the model
    pretrained_weights_path = "/home/maoshufan/JIA2023819/pretrain_weight/efficientnet-b0-355c32eb.pth"  # Replace with your model weights file path
    model = CustomEfficientNet(weights_path=pretrained_weights_path, pretrained=True, freeze=False)

    # Load the saved model weights
    model.load_state_dict(torch.load("best_metric_model_classification2d_array.pth"))  # Replace with your saved model file path
    model = model.to('cuda')  # Replace 'cuda' with 'cpu' if you are not using a GPU

    # Define the loss function
    loss_function = torch.nn.CrossEntropyLoss()

    # Generate validation results
    thresholds = np.arange(0.2, 0.55, 0.005)  # This generates values from 0.01 to 0.99 with a step of 0.01
    for threshold in thresholds:
        val_loss, auc, metric = eval_model(model, val_loader, loss_function, 'cuda', threshold=threshold)  # Replace 'cuda' with 'cpu' if you are not using a GPU
        print(f"Threshold: {threshold:.4f}, Validation loss: {val_loss:.4f}, AUC: {auc:.4f}, Accuracy: {metric:.4f}")

    print(f"Validation loss: {val_loss:.4f}, AUC: {auc:.4f}, Accuracy: {metric:.4f}")

if __name__ == "__main__":
    main()