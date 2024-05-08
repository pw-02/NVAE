import tensorflow as tf
import os
from PIL import Image

# Function to create a TFRecord from image paths and labels
def create_tfrecord(output_filename, image_paths, labels):
    with tf.io.TFRecordWriter(output_filename) as writer:
        for img_path, label in zip(image_paths, labels):
            # Read image data
            with open(img_path, 'rb') as img_file:
                img_data = img_file.read()

            # Create TFRecord features for the image and label
            feature = {
                'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_data])),
                'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label.encode('utf-8')]))  # Use string for labels
            }

            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())

# Set the base directory for the dataset
base_dir = "C:\\Users\\pw\\projects\\datasets\\ImageNet\\imagenet"
train_dir = os.path.join(base_dir, "train")

# Create TFRecord file for the train partition
output_file = "tiny_imagenet_train.tfrecord"

# Initialize lists for image paths and labels
image_paths = []
labels = []

# Traverse the directory structure to get image paths and grandparent folder labels
for root, _, files in os.walk(train_dir):
    for file in files:
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):  # Add more extensions if needed
            image_path = os.path.join(root, file)
            image_paths.append(image_path)

            # Extract the grandparent folder name as the label
            grandparent_folder = os.path.basename(os.path.dirname(os.path.dirname(image_path)))
            labels.append(grandparent_folder)

# Create the TFRecord file with the grandparent folder as the label
create_tfrecord(output_file, image_paths, labels)
