from fileutility import extract_largest_masks, extract_largest_mask_and_neighbors_with_crop,extract_and_save_cropped_images
def main():
    image_dir = "/data/maoshufan/jiadata/T2WI-1/images_adc/"
    mask_dir = "/data/maoshufan/jiadata/T2WI-1/masks_adc/"
    output_dir = "/data/maoshufan/jiadata/T2WI-1/images_adc_crop"
    extract_and_save_cropped_images(image_dir, mask_dir, output_dir, crop_size=(128, 128, 3))

if __name__ == "__main__":
    main()