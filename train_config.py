# train_config.py
import torch
import time

class TrainConfig:
    def __init__(self):
        # File paths and column numbers for data extraction
        self.file_path_adc = "/data/maoshufan/jiadata/T2WI-1/images_adc_crop/"
        self.file_path_T2W1 = '/data/maoshufan/jiadata/T2WI-1/images_cropxy_full'
        self.csv_file_labels = "/data/maoshufan/jiadata/T2WI-1/result.csv"
        self.save_path = "/data/maoshufan/jiadata/T2WI-1/imagesave/"
        self.column_a = 1
        self.column_m = 13

        # Training parameters
        self.batch_size = 8
        self.num_workers = 2
        self.pin_memory = torch.cuda.is_available()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.weights = [0.7, 0.3]  # adjust as needed
        self.learning_rate = 5e-4
        self.learning_rate_expanded = 5e-3
        self.val_interval = 1
        self.max_epochs = 300

        # CSV file for storing training data
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.csv_file = f'training_data_{timestamp}.csv'