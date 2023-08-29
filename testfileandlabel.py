import fileutility
from collections import Counter

def test_generate_file_and_label_lists_from_extracted_images():
    nii_dir = '/data/maoshufan/jiadata/T2WI-1/image_save_near/'
    csv_file = '/data/maoshufan/jiadata/T2WI-1/result.csv'

    file_list, label_list = fileutility.generate_file_and_label_lists_from_extracted_images(nii_dir, csv_file)

    print("File List:")
    for file in file_list:
        print(file)

    print("\nLabel List:")
    for label in label_list:
        print(label)

    # Count the number of 1s, 2s, and 3s in the label list
    label_list = [int(label) for label in label_list]
    label_counts = Counter(label_list)
    print("\nLabel Counts:")
    for i in [1, 2, 3]:
        print(f"Count of {i}: {label_counts[i]}")

if __name__ == "__main__":
    test_generate_file_and_label_lists_from_extracted_images()