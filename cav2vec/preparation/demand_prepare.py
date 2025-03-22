# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import tempfile
import shutil
import os, sys, subprocess, glob, re
import numpy as np
from collections import defaultdict
from scipy.io import wavfile
from tqdm import tqdm

def split_demand(demand_root, rank, nshard):
    wav_fns = glob.glob(f"{demand_root}/*/*wav")
    num_per_shard = math.ceil(len(wav_fns)/nshard)
    start_id, end_id = num_per_shard*rank, num_per_shard*(rank+1)
    wav_fns = wav_fns[start_id: end_id]
    print(f"{len(wav_fns)} raw audios")
    output_dir = f"{demand_root}/short-demand"
    dur = 10
    for wav_fn in tqdm(wav_fns):
        sample_rate, wav_data = wavfile.read(wav_fn)
        assert sample_rate == 16_000 and len(wav_data.shape) == 1
        if len(wav_data) > dur * sample_rate:
            num_split = int(np.ceil(len(wav_data) / (dur*sample_rate)))
            for i in range(num_split):
                filename = '/'.join(wav_fn.split('/')[-2:])[:-4]
                output_wav_fn = os.path.join(output_dir, filename + f'-{i}.wav')
                sub_data = wav_data[i*dur*sample_rate: (i+1)*dur*sample_rate]
                os.makedirs(os.path.dirname(output_wav_fn), exist_ok=True)
                wavfile.write(output_wav_fn, sample_rate, sub_data.astype(np.int16))
    return

def count_frames(wav_fns, rank, nshard):
    num_per_shard = math.ceil(len(wav_fns)/nshard)
    start_id, end_id = num_per_shard*rank, num_per_shard*(rank+1)
    wav_fns = wav_fns[start_id: end_id]
    nfs = []
    for wav_fn in tqdm(wav_fns):
        sample_rate, wav_data = wavfile.read(wav_fn)
        nfs.append(len(wav_data))
    return wav_fns, nfs


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='DEMAND audio preparation', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--demand', type=str, help='DEMAND root')
    parser.add_argument('--nshard', type=int, default=1, help='number of shards')
    parser.add_argument('--rank', type=int, default=0, help='rank id')
    args = parser.parse_args()

    tmp_dir = tempfile.mkdtemp(dir='./')
    ranks = list(range(0, args.nshard))
    print(f"Split raw audio")
    split_demand(args.demand, args.rank, args.nshard)
    short_demand = os.path.join(args.demand, 'short-demand')
    print(f"Count number of frames")
    wav_fns = glob.glob(f"{short_demand}/*/*wav")
    wav_fns, nfs = count_frames(wav_fns, args.rank, args.nshard)
    assert len(nfs) == len(wav_fns)
    num_frames_fn = f"{short_demand}/nframes.audio"
    with open(num_frames_fn, 'a+') as fo:
        for wav_fn, nf in zip(wav_fns, nfs):
            fo.write(os.path.abspath(wav_fn)+'\t'+str(nf)+'\n')
    shutil.rmtree(tmp_dir)
