import sys, os, glob, subprocess, shutil, math
from datetime import timedelta
import tempfile
from collections import OrderedDict
from pydub import AudioSegment
from tqdm import tqdm


def read_csv(csv_file, delimit=','):
    lns = open(csv_file, 'r').readlines()
    keys = lns[0].strip().split(delimit)
    df = {key: [] for key in keys}
    for ln in lns[1:]:
        ln = ln.strip().split(delimit)
        for j, key in enumerate(keys):
            df[key].append(ln[j])
    return df


def trim_pretrain(root_dir, output_dir, ffmpeg, rank=0, nshard=1, split='val'):
    pretrain_dir = os.path.join(root_dir, 'v2', 'clips')
    print(f"Trim original videos in pretrain")
    csv_fn = os.path.join(output_dir, f'ref.{split}.csv')

    print(f"Step 1. Trim video and audio")
    output_video_dir, output_audio_dir = os.path.join(output_dir, f'short-{split}'), os.path.join(output_dir, f'audio/short-{split}')
    os.makedirs(output_video_dir, exist_ok=True)
    os.makedirs(output_audio_dir, exist_ok=True)
    trim_video_frame(csv_fn, pretrain_dir, output_video_dir, ffmpeg, rank, nshard)
    trim_audio(csv_fn, pretrain_dir, output_audio_dir, ffmpeg, rank, nshard)

    return

def trim_video_frame(csv_fn, raw_dir, output_dir, ffmpeg, rank, nshard):
    df = read_csv(csv_fn)
    raw2fid = OrderedDict()
    decimal, fps = 9, 30
    for fid, start, end in zip(df['clip'], df['tb'], df['te']):
        if '_' in fid:
            raw_fid = '_'.join(fid.split('_')[:-1])
        else:
            raw_fid = fid
        if raw_fid in raw2fid:
            raw2fid[raw_fid].append([fid, start, end])
        else:
            raw2fid[raw_fid] = [[fid, start, end]]
    i_raw = -1
    num_per_shard = math.ceil(len(raw2fid.keys())/nshard)
    start_id, end_id = num_per_shard*rank, num_per_shard*(rank+1)
    fid_info_shard = list(raw2fid.items())[start_id: end_id]
    print(f"Total videos in current shard: {len(fid_info_shard)}/{len(raw2fid.keys())}")
    for raw_fid, fid_info in tqdm(fid_info_shard):
        i_raw += 1
        raw_path = os.path.join(raw_dir, raw_fid+'.mp4')
        tmp_dir = tempfile.mkdtemp()
        cmd = ffmpeg + " -i " + raw_path + " " + tmp_dir + '/%0' + str(decimal) + 'd.png -loglevel quiet'
        subprocess.call(cmd, shell=True)
        num_frames = len(glob.glob(tmp_dir+'/*png'))
        for fid, start_sec, end_sec in fid_info:
            sub_dir = os.path.join(tmp_dir, fid)
            os.makedirs(sub_dir, exist_ok=True)
            start_sec, end_sec = float(start_sec), float(end_sec)
            if end_sec == -1:
                end_sec = 24*3600
            start_frame_id, end_frame_id = int(start_sec*fps), min(int(end_sec*fps), num_frames)
            imnames = [tmp_dir+'/'+str(x+1).zfill(decimal)+'.png' for x in range(start_frame_id, end_frame_id)]
            for ix, imname in enumerate(imnames):
                shutil.copyfile(imname, sub_dir+'/'+str(ix).zfill(decimal)+'.png')
            output_path = os.path.join(output_dir, fid+'.mp4')
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cmd = [ffmpeg, "-framerate", "30", "-i", sub_dir+'/%0'+str(decimal)+'d.png', "-y", "-crf", "20", "-r", "25", output_path, "-loglevel", "quiet"]  # input frame rate: 25 -> output frame rate: 30

            pipe = subprocess.call(cmd, stdout = subprocess.PIPE, stderr = subprocess.STDOUT) # subprocess.PIPE
        shutil.rmtree(tmp_dir)
    return

def trim_audio(csv_fn, raw_dir, output_dir, ffmpeg, rank, nshard):
    df = read_csv(csv_fn)
    raw2fid = OrderedDict()
    for fid, start, end in zip(df['clip'], df['tb'], df['te']):
        if '_' in fid:
            raw_fid = '_'.join(fid.split('_')[:-1])
        else:
            raw_fid = fid
        if raw_fid in raw2fid:
            raw2fid[raw_fid].append([fid, start, end])
        else:
            raw2fid[raw_fid] = [[fid, start, end]]
    i_raw = -1
    num_per_shard = math.ceil(len(raw2fid.keys())/nshard)
    start_id, end_id = num_per_shard*rank, num_per_shard*(rank+1)
    fid_info_shard = list(raw2fid.items())[start_id: end_id]
    print(f"Total audios in current shard: {len(fid_info_shard)}/{len(raw2fid.keys())}")
    for raw_fid, fid_info in tqdm(fid_info_shard):
        i_raw += 1
        tmp_dir = tempfile.mkdtemp()
        wav_path = os.path.join(tmp_dir, 'tmp.wav')
        cmd = ffmpeg + " -i " + os.path.join(raw_dir, raw_fid+'.mp4') + " -ar 16000 -f wav -vn -y " + wav_path + ' -loglevel quiet'
        subprocess.call(cmd, shell=True)
        raw_audio = AudioSegment.from_wav(wav_path)
        for fid, start_sec, end_sec in fid_info:
            start_sec, end_sec = float(start_sec), float(end_sec)
            if end_sec == -1:
                end_sec = 24*3600
            t1, t2 = int(start_sec*1000), int(end_sec*1000)
            new_audio = raw_audio[t1: t2]
            output_path = os.path.join(output_dir, fid+'.wav')
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            new_audio.export(output_path, format="wav")
        shutil.rmtree(tmp_dir)
    return

def get_file_label(output_dir, split='val'):
    print(f"Step 2. Get files and labels")
    video_ids_total, labels_total = [], []
    pretrain_csv = os.path.join(output_dir, f'ref.{split}.csv')
    df = read_csv(pretrain_csv)
    for video_id, label in zip(df['clip'], df['trn']):
        video_ids_total.append(os.path.join(f'short-{split}', video_id))
        labels_total.append(label)
    video_id_fn, label_fn = os.path.join(output_dir, 'file.list'), os.path.join(output_dir, 'label.list')
    print(video_id_fn, label_fn)
    with open(video_id_fn, 'w') as fo:
        fo.write('\n'.join(video_ids_total)+'\n')
    with open(label_fn, 'w') as fo:
        fo.write('\n'.join(labels_total)+'\n')
    return


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ego4d', type=str, help='ego4d root dir')
    parser.add_argument('--output-dir', type=str, help='output dir')
    parser.add_argument('--split', type=str, help='{train | val}')
    parser.add_argument('--ffmpeg', type=str, help='path to ffmpeg')
    parser.add_argument('--rank', type=int, help='rank id')
    parser.add_argument('--nshard', type=int, help='number of shards')
    parser.add_argument('--step', type=int, help='Steps')
    args = parser.parse_args()
    if args.step == 1:
        trim_pretrain(args.ego4d, args.output_dir, args.ffmpeg, args.rank, args.nshard, args.split)
    elif args.step == 2:
        get_file_label(args.output_dir, args.split)