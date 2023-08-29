import os
import csv
import nibabel as nib
import numpy as np

def generate_file_and_label_lists(nii_dir, csv_file):
    # Read the CSV file into a dictionary
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        labels_dict = {rows[0]:rows[1] for rows in reader}

    file_list = []
    label_list = []

    # Read the .nii.gz files and match with labels
    for filename in sorted(os.listdir(nii_dir)):
        if filename.endswith('.nii.gz'):
            patient_number = filename.split('.')[0]  # Remove the .nii.gz extension
            if patient_number in labels_dict:
                file_list.append(os.path.join(nii_dir, filename))
                label_list.append(labels_dict[patient_number])
            else:
                print(f'Patient number: {patient_number} not found in CSV file.')
    
    return file_list, label_list

def generate_file_and_label_lists_from_extracted_images(nii_dir, csv_file):
    # Read the CSV file into a dictionary
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        labels_dict = {rows[0]:rows[1] for rows in reader}

    file_list = []
    label_list = []

    # Read the .nii.gz files and match with labels
    for filename in sorted(os.listdir(nii_dir)):
        if filename.endswith('.nii.gz'):
            # Extract the patient number from the new filename structure
            patient_number = filename.split('_')[-1].split('.')[0]  # Remove the 'extracted_image_' prefix and '.nii.gz' extension
            if patient_number in labels_dict:
                file_list.append(os.path.join(nii_dir, filename))
                label_list.append(labels_dict[patient_number])
            else:
                print(f'Patient number: {patient_number} not found in CSV file.')
    
    return file_list, label_list

def generate_image_and_mask_pairs(image_dir, mask_dir):
    image_list = []
    mask_list = []

    # Read the .nii.gz files and match with masks
    for filename in sorted(os.listdir(image_dir)):
        if filename.endswith('.nii.gz'):
            image_path = os.path.join(image_dir, filename)
            mask_path = os.path.join(mask_dir, filename)
            if os.path.exists(mask_path):
                image_list.append(image_path)
                mask_list.append(mask_path)
            else:
                print(f'Mask for image: {filename} not found.')
    
    return image_list, mask_list

def read_3d_masks(mask_list, num_masks=3):
    masks = []
    z_indices = []

    for mask_path in mask_list:
        mask = nib.load(mask_path).get_fdata()
        masks.append(mask)

        # Calculate the sum of each 2D mask in the z-axis
        mask_sizes = np.sum(mask, axis=(0, 1))

        # Find the indices of the largest 2D masks
        largest_masks_indices = np.argsort(mask_sizes)[-num_masks:]
        z_indices.append(largest_masks_indices)

    return masks, z_indices

def save_image(image, output_path):
    img_nifti = nib.Nifti1Image(image, np.eye(4))
    nib.save(img_nifti, output_path)

def extract_largest_masks(image_dir, mask_dir, output_dir, num_masks=3):
    # Generate image and mask pairs
    image_list, mask_list = generate_image_and_mask_pairs(image_dir, mask_dir)

    # Read 3D masks and get z-indices of largest 2D masks
    masks, z_indices = read_3d_masks(mask_list, num_masks)

    # Extract the corresponding 2D images and masks
    for i, image_path in enumerate(image_list):
        image = nib.load(image_path).get_fdata()

        # Initialize an empty 3D array to store the 2D slices
        image_3d = np.empty((image.shape[0], image.shape[1], num_masks))
        mask_3d = np.empty((image.shape[0], image.shape[1], num_masks))  # For storing the masks

        for j, z_index in enumerate(z_indices[i]):
            # Extract the 2D image
            image_2d = image[:, :, z_index]

            # Extract the 2D mask
            mask_2d = masks[i][:, :, z_index]

            # Add the 2D image and mask to the 3D arrays
            image_3d[:, :, j] = image_2d
            mask_3d[:, :, j] = mask_2d

        # Extract the original image ID from the image path
        original_image_id = os.path.basename(image_path).split('.')[0]

        # Save the extracted 3D image and mask as new .nii.gz files
        output_image_path = os.path.join(output_dir, f'extracted_image_{original_image_id}.nii.gz')
        output_mask_path = os.path.join(output_dir, f'extracted_mask_{original_image_id}.nii.gz')  # Path for the mask
        save_image(image_3d, output_image_path)
        save_image(mask_3d, output_mask_path)  # Save the mask
        return output_image_path, output_mask_path

def read_3d_mask_single(mask_list):
    masks = []
    z_indices = []

    for mask_path in mask_list:
        mask = nib.load(mask_path).get_fdata()
        masks.append(mask)

        # Calculate the sum of each 2D mask in the z-axis
        mask_sizes = np.sum(mask, axis=(0, 1))

        # Find the index of the largest 2D mask
        largest_mask_index = np.argmax(mask_sizes)
        z_indices.append(largest_mask_index)

    return masks, z_indices

def extract_largest_mask_and_neighbors(image_dir, mask_dir, output_dir):

    # Generate image and mask pairs
    image_list, mask_list = generate_image_and_mask_pairs(image_dir, mask_dir)

    # Read 3D masks and get z-index of largest 2D mask
    masks, z_indices = read_3d_mask_single(mask_list)

    # Extract the corresponding 2D images and masks
    for i, image_path in enumerate(image_list):
        image = nib.load(image_path).get_fdata()

        # Initialize an empty 3D array to store the 2D slices
        image_3d = np.empty((image.shape[0], image.shape[1], 3))
        mask_3d = np.empty((image.shape[0], image.shape[1], 3))  # For storing the masks

        for j in range(-1, 2):
            z_index = z_indices[i] + j

            # Ensure z_index is within the valid range
            z_index = max(0, min(z_index, image.shape[2] - 1))

            # Extract the 2D image
            image_2d = image[:, :, z_index]

            # Extract the 2D mask
            mask_2d = masks[i][:, :, z_index]

            # Add the 2D image and mask to the 3D arrays
            image_3d[:, :, j+1] = image_2d
            mask_3d[:, :, j+1] = mask_2d

        # Extract the original image ID from the image path
        original_image_id = os.path.basename(image_path).split('.')[0]

        # Save the extracted 3D image and mask as new .nii.gz files
        output_image_path = os.path.join(output_dir, f'extracted_image_{original_image_id}.nii.gz')
        output_mask_path = os.path.join(output_dir, f'extracted_mask_{original_image_id}.nii.gz')  # Path for the mask
        save_image(image_3d, output_image_path)
        save_image(mask_3d, output_mask_path)  # Save the mask

def read_3d_mask_single_with_center(mask_list):
    masks = []
    z_indices = []
    centers = []

    for mask_path in mask_list:
        mask = nib.load(mask_path).get_fdata()
        masks.append(mask)

        # Calculate the sum of each 2D mask in the z-axis
        mask_sizes = np.sum(mask, axis=(0, 1))

        # Find the index of the largest 2D mask
        largest_mask_index = np.argmax(mask_sizes)
        z_indices.append(largest_mask_index)

        # Find the center of the largest 2D mask
        largest_mask = mask[:, :, largest_mask_index]
        center_x = np.mean(np.where(largest_mask > 0)[0])
        center_y = np.mean(np.where(largest_mask > 0)[1])
        centers.append((center_x, center_y))

    return masks, z_indices, centers

def extract_largest_mask_and_neighbors_with_crop(image_dir, mask_dir, output_dir):
    # Generate image and mask pairs
    image_list, mask_list = generate_image_and_mask_pairs(image_dir, mask_dir)

    # Read 3D masks and get z-index of largest 2D mask
    masks, z_indices, centers = read_3d_mask_single_with_center(mask_list)

    # Extract the corresponding 2D images and masks
    for i, image_path in enumerate(image_list):
        image = nib.load(image_path).get_fdata()

        # Initialize an empty 3D array to store the 2D slices
        image_3d = np.empty((128, 128, 3))
        mask_3d = np.empty((128, 128, 3))  # For storing the masks

        for j in range(-1, 2):
            z_index = z_indices[i] + j

            # Ensure z_index is within the valid range
            z_index = max(0, min(z_index, image.shape[2] - 1))

            # Crop the 2D image and mask around the center
            center_x, center_y = centers[i]
            start_x = int(max(0, center_x - 64))
            end_x = int(min(image.shape[0], center_x + 64))
            start_y = int(max(0, center_y - 64))
            end_y = int(min(image.shape[1], center_y + 64))

            image_2d = image[start_x:end_x, start_y:end_y, z_index]
            mask_2d = masks[i][start_x:end_x, start_y:end_y, z_index]

            # Add the 2D image and mask to the 3D arrays
            image_3d[:, :, j+1] = image_2d
            mask_3d[:, :, j+1] = mask_2d

        # Extract the original image ID from the image path
        original_image_id = os.path.basename(image_path).split('.')[0]

        # Save the extracted 3D image and mask as new .nii.gz files
        output_image_path = os.path.join(output_dir, f'extracted_image_{original_image_id}.nii.gz')
        output_mask_path = os.path.join(output_dir, f'extracted_mask_{original_image_id}.nii.gz')  # Path for the mask
        save_image(image_3d, output_image_path)
        save_image(mask_3d, output_mask_path)  # Save the mask
        