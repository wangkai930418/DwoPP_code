import torch
import glob
import re
import numpy as np
import torch.nn.functional as F


def euclidean_metric(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = -((a - b) ** 2).sum(dim=2)
    return logits


def decode(proto, data_query, temperature=1):
    logits = (
            euclidean_metric(data_query, proto)
            / temperature
    )
    return logits


def cosine_decode(proto, data_query, temperature=1):
    proto=F.normalize(proto)
    data_query=F.normalize(data_query)
    logits = (torch.mm(data_query,proto.t())/ temperature)
    
    return logits

def float_or_string(arg):

    try:
        return float(arg)
    except (ValueError, TypeError):
        return arg


def flip_img(img):

    inv_idx = torch.arange(img.size(3)-1, -1, -1).long()  # N x C x H x W
    img_flipped = img.index_select(3, inv_idx)
    return img_flipped


def get_id(img_path, dataset):

    camera_id = []
    labels = []
    for path in img_path:
        filename = path.split('/')[-1]
        label = filename[0:4]
        camera = filename.split('c')[1]
        if label[0:2] == '-1':
            labels.append(-1)
        else:
            labels.append(int(label))
        camera_id.append(int(camera[0]))
    return camera_id, labels

def get_label_camera(dataset, imgs):
    camera_id = []
    labels = []
    for path in imgs:
        filename = path.split('/')[-1]
        camera = dataset.camera(filename)
        camera_id.append(int(camera))
        label = dataset.id(filename)
        labels.append(int(label))
    return camera_id, labels

def get_camera(dataset, imgs):
    camera_id = []
    for path in imgs:
        filename = path.split('/')[-1]
        camera = dataset.camera(filename)
        camera_id.append(int(camera))
    return camera_id


def compute_map(index, good_index, junk_index):

    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()
    if good_index.size==0:   # if empty
        cmc[0] = -1
        return ap, cmc

    # remove junk_index
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]

    # find good_index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask==True)
    rows_good = rows_good.flatten()
    
    cmc[rows_good[0]:] = 1
    for i in range(ngood):
        d_recall = 1.0 / ngood
        precision = (i+1) * 1.0 / (rows_good[i] + 1)
        if rows_good[i] != 0:
            old_precision = i * 1.0 / rows_good[i]
        else:
            old_precision = 1.0
        ap = ap + d_recall * (old_precision + precision) / 2

    return ap, cmc


def get_label_camera(dataset, imgs):
    camera_id = []
    labels = []
    for path in imgs:
        filename = path.split('/')[-1]
        camera = dataset.camera(filename)
        camera_id.append(int(camera))
        label = dataset.id(filename)
        labels.append(int(label))
    return camera_id, labels

def get_camera(dataset, imgs):
    camera_id = []
    for path in imgs:
        filename = path.split('/')[-1]
        camera = dataset.camera(filename)
        camera_id.append(int(camera))
    return camera_id