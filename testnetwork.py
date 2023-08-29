import torch
from modelresnet import ResNetModel
def main():
    # Instantiate the model
    num_categories = 10  # for example, 10 categories
    num_channels = 3  # for example, RGB images
    model = ResNetModel(num_categories, num_channels)

    # Generate a random tensor with the correct dimensions
    # For example, a batch of 4 images, each with 3 channels (RGB), and each image is 224x224
    input_tensor = torch.randn(4, num_channels, 224, 224)

    # Forward pass through the model
    output_tensor = model(input_tensor)

    # Print the output
    print(output_tensor)

if __name__ == "__main__":
    main()