# ---------------------------------------------------------------
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for NVAE. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import torch.utils.data as data
import numpy as np
import lmdb
import os
import io
from PIL import Image
import os
import os.path as osp
import pickle
import six

def num_samples(dataset, train):
    if dataset == 'celeba':
        return 27000 if train else 3000
    elif dataset == 'celeba64':
        return 162770 if train else 19867
    elif dataset == 'imagenet-oord':
        return 1281147 if train else 50000
    elif dataset == 'ffhq':
        return 63000 if train else 7000
    elif dataset == 'metfaces':
        return 1336 if train else 300
    else:
        raise NotImplementedError('dataset %s is unknown' % dataset)

def loads_data(buf):
    """
    Args:
        buf: the output of `dumps`.
    """
    return pickle.loads(buf)

class LMDBDataset(data.Dataset):
    def __init__(self, root, name='', train=True, transform=None, is_encoded=False):
        self.train = train
        self.name = name
        self.transform = transform
        if self.train:
            lmdb_path = os.path.join(root, 'train.lmdb')
        else:
            lmdb_path = os.path.join(root, 'validation.lmdb')
        self.data_lmdb = lmdb.open(lmdb_path, subdir=osp.isdir(lmdb_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.data_lmdb.begin(write=False) as txn:
            self.length = loads_data(txn.get(b'__len__'))
            self.keys = loads_data(txn.get(b'__keys__'))
        # self.data_lmdb = lmdb.open(lmdb_path, readonly=True, max_readers=1, lock=False, readahead=False, meminit=False)
        
        self.is_encoded = is_encoded
    def __getitem__(self, index):
        target = [0]
        with self.data_lmdb.begin(write=False, buffers=True) as txn:
            data = txn.get(self.keys[index])
        unpacked = loads_data(data)
            # load img
        imgbuf = unpacked[0]
        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        img = Image.open(buf).convert('RGB')

        # load label
        target = unpacked[1]
        # if self.is_encoded:
        #     img = Image.open(io.BytesIO(data))
        #     img = img.convert('RGB')
        # else:
        #     img = np.asarray(data, dtype=np.uint8)
        #     # assume data is RGB
        #     size = int(np.sqrt(len(img) / 3))
        #     img = np.reshape(img, (size, size, 3))
        #     img = Image.fromarray(img, mode='RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, target
    # def __getitem__(self, index):
    #     target = [0]
    #     with self.data_lmdb.begin(write=False, buffers=True) as txn:
    #         data = txn.get(str(index).encode())
    #         if self.is_encoded:
    #             img = Image.open(io.BytesIO(data))
    #             img = img.convert('RGB')
    #         else:
    #             img = np.asarray(data, dtype=np.uint8)
    #             # assume data is RGB
    #             size = int(np.sqrt(len(img) / 3))
    #             img = np.reshape(img, (size, size, 3))
    #             img = Image.fromarray(img, mode='RGB')

    #     if self.transform is not None:
    #         img = self.transform(img)

    #     return img, target

    def __len__(self):
        return  self.length
        # return num_samples(self.name, self.train)
