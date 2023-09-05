import random
import pickle
from fileutility import generate_file_and_label_lists_from_extracted_images

def save_to_file(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def load_from_file(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)
    
def generate_balanced_sets(file_list, label_list, train_size=250, val_size=50):
    # Separate positive and negative samples
    label_list = [0 if int(label) in [1, 2] else 1 for label in label_list]

    pos_samples = [(file, label) for file, label in zip(file_list, label_list) if label == 1]
    neg_samples = [(file, label) for file, label in zip(file_list, label_list) if label == 0]

    # Shuffle the samples
    random.shuffle(pos_samples)
    random.shuffle(neg_samples)

    # Create balanced validation set
    half_val_size = val_size // 2
    val_set = pos_samples[:half_val_size] + neg_samples[:half_val_size]
    random.shuffle(val_set)  # Shuffle the validation set

    # Create training set
    train_set = pos_samples[half_val_size:half_val_size+train_size] + neg_samples[half_val_size:half_val_size+train_size]
    random.shuffle(train_set)  # Shuffle the training set

    # Separate files and labels
    train_files, train_labels = zip(*train_set)
    val_files, val_labels = zip(*val_set)

    return list(train_files), list(train_labels), list(val_files), list(val_labels)

if __name__ == "__main__":
    file_path = "/data/maoshufan/jiadata/T2WI-1/images_adc_crop"
    csv_file_labels = "/data/maoshufan/jiadata/T2WI-1/result.csv"
    file_list, label_list = generate_file_and_label_lists_from_extracted_images(file_path,csv_file_labels)
    train_files, train_labels, val_files, val_labels = generate_balanced_sets(file_list, label_list)
    # Save the train and validation sets to files
    save_to_file((train_files, train_labels), 'train_set_ADC.pkl')
    save_to_file((val_files, val_labels), 'val_set_ADC.pkl')
