import torch.utils.data as Data
from torchvision.datasets.folder import default_loader

from os.path import join as ospj

from data.common import list_pictures
from torchvision import transforms
from torch.utils.data import dataloader

import numpy as np
import random
import pickle as pkl
import os


class Market1501(Data.Dataset):
    def __init__(self, dataset_root, transform, task_id=0, split='train',
                 ROOT_PATH='preprocess_dataset/',dataset_name='market1501'):

        if not split in ['train', 'gallery', 'query']:
            raise Exception('Invalid dataset split.')
        self.transform = transform
        self.loader = default_loader
        self.split = split
        self.dataset_root=dataset_root

        if dataset_name=='market1501':
            BASE_CLS_NUM = 76
            TASK_NUM = 10

        if split == 'train':
            file_name = '_'.join(('base', str(BASE_CLS_NUM), 'task', str(TASK_NUM))) + '.pkl'

            with open(ospj(ROOT_PATH, dataset_name, file_name), 'rb') as h:
                CL_data_split = pkl.load(h)

            curr_task_split=CL_data_split[task_id]
            self.train_label= curr_task_split['train_label']
            self.train_data=curr_task_split['train_data']
            print(f'load {task_id} train data done!')

        else:
            if split == 'gallery':
                data_path = ospj(dataset_root, 'bounding_box_test')
            elif split == 'query':
                data_path = ospj(dataset_root, 'query')

            self.imgs = [path for path in list_pictures(data_path)]


    def __getitem__(self, index):
        if self.split == 'train':
            ref_path = self.train_data[index]
            path=os.path.join(self.dataset_root,ref_path)
            label=self.train_label[index]

        else:
            path = self.imgs[index]
            label = self._id2label[self.id(path)] if self.split == 'train' else self.id(path)

        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        if self.split=='train':
            return len(self.train_label)
        else:
            return len(self.imgs)

    @staticmethod
    def id(file_path):
        """
        file_path: unix style file path
        return: person id
        """
        return int(file_path.split('/')[-1].split('_')[0])

    @staticmethod
    def camera(file_path):
        """
        file_path: unix style file path
        return: camera id
        """
        return int(file_path.split('/')[-1].split('_')[1][1])

    @property
    def ids(self):
        """
        return: person id list corresponding to dataset image paths
        """
        return [self.id(path) for path in self.imgs]

    @property
    def unique_ids(self):
        """
        return: unique person ids in ascending order
        """
        return sorted(set(self.ids))

    @property
    def cameras(self):
        """
        return: camera id list corresponding to dataset image paths
        """
        return [self.camera(path) for path in self.imgs]
