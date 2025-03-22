# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math, time
import os, sys, subprocess, glob, re
import numpy as np
from collections import defaultdict
from scipy.io import wavfile
from tqdm import tqdm

DEMAND_ENVIRONMENTS = ['DWASHING', 
                       'DKITCHEN', 
                       'TCAR', 
                       'TBUS', 
                       'OMEETING', 
                       'SPSQUARE', 
                       'NFIELD', 
                       'NPARK', 
                       'NRIVER', 
                       'OHALLWAY', 
                       'TMETRO', 
                       'OOFFICE', 
                       'DLIVING', 
                       'PRESTO', 
                       'STRAFFIC', 
                       'SCAFE', 
                       'PSTATION',
                       'PCAFETER'
                    ]

def make_demand_tsv(demand_root, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    sample_rate = 16_000
    min_dur, max_dur = 3*sample_rate, 11*sample_rate
    all_fns = {}
    nfs = f"{demand_root}/nframes.audio"
    nfs = dict([x.strip().split('\t') for x in open(nfs).readlines()])
    for category in DEMAND_ENVIRONMENTS:
        wav_fns = glob.glob(f"{demand_root}/{category}/*wav")
        target_fns = []
        for wav_fn in tqdm(wav_fns):
            dur = int(nfs[os.path.abspath(wav_fn)])
            if dur >= min_dur and dur < max_dur:
                target_fns.append(wav_fn)
        print(f"{category}: {len(target_fns)}/{len(wav_fns)}")
        all_fns[category] = target_fns
        output_subdir = f"{output_dir}/{category}"
        os.makedirs(output_subdir, exist_ok=True)
        # only test sets
        np.random.shuffle(target_fns)
        test_fns = target_fns
        test_fns = [os.path.abspath(test_fn) for test_fn in test_fns]
        print(os.path.abspath(output_subdir), 'test', len(test_fns))
        with open(f"{output_subdir}/test.tsv", 'w') as fo:
            fo.write('\n'.join(test_fns)+'\n')
    return

def combine(input_tsv_dirs, output_dir):
    output_subdir = f"{output_dir}/all"
    os.makedirs(output_subdir, exist_ok=True)
    num_train_per_cat = 20_000
    train_fns, valid_fns, test_fns = [], [], []
    for input_tsv_dir in input_tsv_dirs:
        test_fn = [ln.strip() for ln in open(f"{input_tsv_dir}/test.tsv").readlines()]
        test_fns.extend(test_fn)
    print(os.path.abspath(output_subdir), 'tset', len(test_fns))
    with open(f"{output_subdir}/test.tsv", 'w') as fo:
        fo.write('\n'.join(test_fns)+'\n')
    return

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Set up noise manifest', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--demand', type=str, help='demand root')
    args = parser.parse_args()
    short_demand, output_tsv_dir = f"{args.demand}/short-demand", f"{args.demand}/tsv"
    print(f"Make tsv files")
    make_demand_tsv(short_demand, output_tsv_dir)
    print(f"Combine tsv")
    input_tsv_dirs = [f"{output_tsv_dir}/{x}" for x in DEMAND_ENVIRONMENTS]
    combine(input_tsv_dirs, output_tsv_dir)
    return


if __name__ == '__main__':
    main()
