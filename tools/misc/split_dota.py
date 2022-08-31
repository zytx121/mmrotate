# Copyright (c) OpenMMLab. All rights reserved.
import glob
import argparse
import os.path as osp
import shutil
import numpy as np
from mmengine.fileio import dump, load
from mmengine.utils import mkdir_or_exist, track_parallel_progress

prog_description = '''K-Fold dota split.

To split dota data for semi-supervised object detection:
    python tools/misc/split_dota.py
'''


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data-root',
        type=str,
        help='The data root of coco dataset.',
        default='/cluster/home/it_stu7/main/datasets/split_ss_dota/trainval/')
    parser.add_argument(
        '--labeled-percent',
        type=float,
        nargs='+',
        help='The percentage of labeled data in the training set.',
        default=[1, 2, 5, 10])
    parser.add_argument(
        '--fold',
        type=int,
        help='K-fold cross validation for semi-supervised object detection.',
        default=5)
    args = parser.parse_args()
    return args


def split_dota(data_root, percent, fold):
    """Split dota data for Semi-supervised object detection.

    Args:
        data_root (str): The data root of coco dataset.
        percent (float): The percentage of labeled data in the training set.
        fold (int): The fold of dataset and set as random seed for data split.
    """
    np.random.seed(fold)
    ann_path = osp.join(data_root, 'annfiles')
    img_path = osp.join(data_root, 'images')
    ann_list = glob.glob(osp.join(ann_path, '*.txt'))


    # save labeled and unlabeled
    labeled_name = f'annfiles.{fold}@{percent}'
    unlabeled_name = f'images.{fold}@{percent}-unlabeled'
    labeled_path = osp.join(data_root, labeled_name)
    unlabeled_path = osp.join(data_root, unlabeled_name)
    mkdir_or_exist(labeled_path)
    mkdir_or_exist(unlabeled_path)

    labeled_total = int(percent / 100. * len(ann_list))
    labeled_inds = set(
        np.random.choice(range(len(ann_list)), size=labeled_total))

    for i in range(len(ann_list)):
        img_id = osp.split(ann_list[i])[1][:-4]
        if i in labeled_inds:
            shutil.copy(
            ann_list[i],
            osp.join(labeled_path, img_id + '.txt'))
        else:
            shutil.copy(
            osp.join(img_path, img_id + '.png'),
            osp.join(unlabeled_path, img_id + '.png'))

def multi_wrapper(args):
    return split_dota(*args)


if __name__ == '__main__':
    args = parse_args()
    arguments_list = [(args.data_root, p, f)
                      for f in range(1, args.fold + 1)
                      for p in args.labeled_percent]
    track_parallel_progress(multi_wrapper, arguments_list, args.fold)
