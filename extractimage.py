from fileutility import extract_largest_masks, extract_largest_mask_and_neighbors_with_crop
def main():
    image_dir = "/data/maoshufan/jiadata/T2WI-1/images_adc/"
    mask_dir = "/data/maoshufan/jiadata/T2WI-1/masks_full"
    output_dir = "/data/maoshufan/jiadata/T2WI-1/images_cropxy_full"

    largest_images, largest_masks = extract_largest_mask_and_neighbors_with_crop(image_dir, mask_dir, output_dir)

    print("The paths of the images with the largest masks are:")
    for image_path in largest_images:
        print(image_path)

if __name__ == "__main__":
    main()