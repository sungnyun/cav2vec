# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys, os, glob, subprocess, shutil, math
from datetime import timedelta
import tempfile
from collections import OrderedDict
from pydub import AudioSegment
from tqdm import tqdm


def make_folder(lrs2_root):

    main_dir = os.path.join(lrs2_root, 'main')
    output_trainval_dir = os.path.join(lrs2_root, 'trainval')
    output_test_dir = os.path.join(lrs2_root, 'test')
    
    train = open(os.path.join(lrs2_root, 'train.txt'), 'r')
    val = open(os.path.join(lrs2_root, 'val.txt'), 'r')
    test = open(os.path.join(lrs2_root, 'test.txt'), 'r')
    
    train_lists = train.readlines()
    train_lists = list(set([train_list.strip().split('/')[0] for train_list in train_lists]))
    val_lists = val.readlines()
    val_lists = list(set([val_list.strip().split('/')[0] for val_list in val_lists]))
    test_lists = test.readlines()
    test_lists = list(set([test_list.strip().split('/')[0] for test_list in test_lists]))

    print("copy train list!")
    for train_list in tqdm(train_lists):
        shutil.copytree(os.path.join(main_dir, train_list), os.path.join(output_trainval_dir, train_list))

    print("copy val list!")
    for val_list in tqdm(val_lists):
        shutil.copytree(os.path.join(main_dir, val_list), os.path.join(output_trainval_dir, val_list))

    print("copy test list!")
    for test_list in tqdm(test_lists):
        shutil.copytree(os.path.join(main_dir, test_list), os.path.join(output_test_dir, test_list))

    return



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='LRS2 preprocess pretrain dir', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--lrs2', type=str, help='lrs2 root dir')
    args = parser.parse_args()

    make_folder(args.lrs2)
