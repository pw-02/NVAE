import os
import lmdb
import pickle
from PIL import Image
import numpy as np

# Directory locations
data_dir = "C:\\Users\\pw\\projects\\datasets\\ImageNet\\imagenet"
train_dir = os.path.join(data_dir, "train")
val_dir = os.path.join(data_dir, "val")

# LMDB output directory
lmdb_path = "lmbtest"
os.makedirs(lmdb_path, exist_ok=True)

# Create LMDB environment
new_map_size = 1024 * 1024 * 1024  # 1 GB

env = lmdb.open(lmdb_path, map_size=new_map_size)  # Adjust map_size if needed

# Function to add images and labels to LMDB
def add_to_lmdb(data_list, env, name):
    with env.begin(write=True) as txn:
        for idx, data in enumerate(data_list):
            key = f"{name}_{idx}".encode("ascii")
            txn.put(key, pickle.dumps(data))  # Store serialized data


# Get train data
train_data = []
class_dirs = os.listdir(train_dir)
for class_name in class_dirs:
    class_path = os.path.join(train_dir, class_name, "images")
    for image_name in os.listdir(class_path):
        image_path = os.path.join(class_path, image_name)
        image = Image.open(image_path)
        image_data = np.array(image)
        label = class_name
        train_data.append({"image": image_data, "label": label})

# # Get val data
# val_data = []
# val_annotations_path = os.path.join(val_dir, "val_annotations.txt")
# with open(val_annotations_path, "r") as f:
#     annotations = f.readlines()

# for line in annotations:
#     parts = line.split("\t")
#     image_name = parts[0]
#     label = parts[1]
#     image_path = os.path.join(val_dir, "images", image_name)
#     image = Image.open(image_path)
#     image_data = np.array(image)
#     val_data.append({"image": image_data, "label": label})

# Add data to LMDB
add_to_lmdb(train_data, env, "train")
# add_to_lmdb(val_data, env, "val")

env.close()
print(f"LMDB created at {lmdb_path}")
