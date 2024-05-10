import os
import lmdb
import pickle
from PIL import Image
import numpy as np

# Directory locations
train_dir = "C:\\Users\\pw\\projects\\datasets\\ImageNet\\imagenet\\train"
# LMDB output directory
lmdb_path = "imagenetlmbtest-train"
os.makedirs(lmdb_path, exist_ok=True)

def main():

    # Create LMDB environment
    new_map_size = 1024 * 1024 * 1024 *3 # 1 GB
    env = lmdb.open(lmdb_path, map_size=new_map_size)  # Adjust map_size if needed
    count = 0
    class_dirs = os.listdir(train_dir)
    with env.begin(write=True) as txn:
        for class_name in class_dirs:
            class_path = os.path.join(train_dir, class_name, "images")
            for (idx, image_name) in enumerate(os.listdir(class_path)):
                image_path = os.path.join(class_path, image_name)

                image = Image.open(image_path)
                if image.mode == "L":
                    image = image.convert("RGB")

                image = np.array(image)
                # image = image.resize(size=(128, 128), resample=Image.BILINEAR)
                # image = np.array(image).reshape(image.size[1], image.size[0],3)
                # label = class_name
                txn.put(str(count).encode(), image)
                count += 1
                print(count)
                # if count == 300:
                #     break
           

# # Function to add images and labels to LMDB
# def add_to_lmdb(data_list, env, name):
#     with env.begin(write=True) as txn:
#         for idx, data in enumerate(data_list):
#             key = f"{name}_{idx}".encode("ascii")
#             txn.put(key, pickle.dumps(data))  # Store serialized data

main()
print(f"LMDB created at {lmdb_path}")
