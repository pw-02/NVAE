from PIL import Image
import glob
from PIL import Image
import os

def get_first_five_images(folder_path):
    file_paths = []
    # Traverse the folder structure
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            # Construct the full path to the file
            file_path = os.path.join(root, filename)
            # Add to the list
            file_paths.append(file_path)
            if len(file_path) == 5:
                break
    return file_paths




def resize_and_concat_images(image_paths, output_path, target_size=(128, 128)):
    # Open and resize each image
    resized_images = []
    for image_path in image_paths:
        img = Image.open(image_path)
        img_resized = img.resize(target_size, Image.BILINEAR)
        resized_images.append(img_resized)

    # Calculate the total width for the new image
    total_width = sum(img.width for img in resized_images)

    # Create a new blank image with the required width and the height of the tallest image
    max_height = max(img.height for img in resized_images)
    new_image = Image.new('RGB', (total_width, max_height))

    # Paste each resized image into the new image horizontally
    current_width = 0
    for img in resized_images:
        new_image.paste(img, (current_width, 0))
        current_width += img.width

    # Save the composite image
    new_image.save(output_path)

# Example usage:

# Example usage:
folder_path = "example_imgs\\nvae\\cifar"  # Specify the correct folder path
output_path = "composite_image_nvae_cifar.jpg"

# Get the first five image files from the folder
image_paths = get_first_five_images(folder_path)
resize_and_concat_images(image_paths, output_path)
