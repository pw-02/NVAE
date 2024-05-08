# ---------------------------------------------------------------
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for NVAE. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import argparse
import torch
import numpy as np
import lmdb
import os

from PIL import Image


def main(split, ffhq_img_path, ffhq_lmdb_path):
    assert split in {'train', 'validation'}
    num_images = 1336
    num_train = 1000

    # create target directory
    if not os.path.exists(ffhq_lmdb_path):
        os.makedirs(ffhq_lmdb_path, exist_ok=True)

    ind_path = os.path.join(ffhq_lmdb_path, 'train_test_ind.pt')
    if os.path.exists(ind_path):
        ind_dat = torch.load(ind_path)
        train_ind = ind_dat['train']
        test_ind = ind_dat['test']
    else:
        rand = np.random.permutation(num_images)
        train_ind = rand[:num_train]
        test_ind = rand[num_train:]
        torch.save({'train': train_ind, 'test': test_ind}, ind_path)

    file_ind = train_ind if split == 'train' else test_ind
    lmdb_path = os.path.join(ffhq_lmdb_path, '%s.lmdb' % split)
    new_map_size = 1024 * 1024 * 1024  # 1 GB

    # create lmdb
    env = lmdb.open(lmdb_path, map_size=new_map_size)
    count = 0
    with env.begin(write=True) as txn:
        for file_name in os.listdir(ffhq_img_path):
            # img_path = os.path.join(ffhq_img_path, '%05d.png' % i)
            img_path = os.path.join(ffhq_img_path, file_name)
            im = Image.open(img_path)
            im = im.resize(size=(256, 256), resample=Image.BILINEAR)
            im = np.array(im.getdata(), dtype=np.uint8).reshape(im.size[1], im.size[0], 3)

            txn.put(str(count).encode(), im)
            count += 1
            if count % 100 == 0:
                print(count)

        print('added %d items to the LMDB dataset.' % count)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('FFHQ LMDB creator. Download images1024x1024.zip from here and unzip it \n'
                                     'https://drive.google.com/drive/folders/1WocxvZ4GEZ1DI8dOz30aSj2zT6pkATYS')
    # experimental results
    parser.add_argument('--ffhq_img_path', type=str, default='C:\\Users\\pw\\projects\\datasets\\metfaces-release\\images',
                        help='location of images from FFHQ')
    parser.add_argument('--ffhq_lmdb_path', type=str, default='data\\metfaces-lmdb',
                        help='target location for storing lmdb files')
    parser.add_argument('--split', type=str, default='validation',
                        help='training or validation split', choices=['train', 'validation'])
    args = parser.parse_args()

    main(args.split, args.ffhq_img_path, args.ffhq_lmdb_path)

