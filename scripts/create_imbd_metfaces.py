import os
import lmdb
import pickle
from PIL import Image
import numpy as np

# Directory locations
train_dir = "C:\\Users\\pw\\projects\\datasets\\metfaces-release\\images"
# LMDB output directory
lmdb_path = "metafceslmbtest"
os.makedirs(lmdb_path, exist_ok=True)

def main():

    # Create LMDB environment
    new_map_size = 1024 * 1024 * 1024  # 1 GB
    env = lmdb.open(lmdb_path, map_size=new_map_size)  # Adjust map_size if needed
    count = 0
    with env.begin(write=True) as txn:
        for (idx, image_name) in enumerate(os.listdir(train_dir)):
            image_path = os.path.join(train_dir, image_name)
            image = Image.open(image_path)
            image = np.array(image)
            # image_data = image_data.reshape(image_data.size[1], image_data.size[0], 3)
            # label = class_name
            txn.put(str(idx).encode(), image)
            count += 1
            print(count)
            if count == 300:
                break
           

# # Function to add images and labels to LMDB
# def add_to_lmdb(data_list, env, name):
#     with env.begin(write=True) as txn:
#         for idx, data in enumerate(data_list):
#             key = f"{name}_{idx}".encode("ascii")
#             txn.put(key, pickle.dumps(data))  # Store serialized data

main()
print(f"LMDB created at {lmdb_path}")
