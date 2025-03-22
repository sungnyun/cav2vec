# Data Preprocessing

This folder contains scripts for data preparation for LRS3 and VoxCeleb2 datasets, as well as audio noise preparation (for noisy environment simulation).

## Installation
To preprocess, you need some additional packages:
```
pip install cmake  # if installing dlib causes error    
pip install -r requirements.txt
```

## LRS3 Preprocessing

Download and decompress the [data](https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs3.html). Assume the data directory is `${lrs3}`, which contains three folders (`pretrain,trainval,test`). Follow the steps below:

### 1. Data preparation
/path/to/ffmpeg = (which ffmpeg)
```sh
python lrs3_prepare.py --lrs3 ${lrs3} --ffmpeg /path/to/ffmpeg --rank ${rank} --nshard ${nshard} --step ${step}
```
This will generate a list of file-ids (`${lrs3}/file.list`) and corresponding text labels (`${lrs3}/label.list`). Specifically, it includes 4 steps, where `${step}` ranges from `1,2,3,4`. Step 1, split long utterances in LRS3 `pretraining` into shorter utterances, generate their time boundaries and labels. Step 2, trim videos and audios according to the new time boundary. Step 3, extracting audio for trainval and test split. Step 4, generate a list of file ids and corresponding text transcriptions.  `${nshard}` and `${rank}` are only used in step 2 and 3. This would shard all videos into `${nshard}` and processes `${rank}`-th shard, where rank is an integer in `[0,nshard-1]`. 


### 2. Detect facial landmark and crop mouth ROIs:
```sh
python detect_landmark.py --root ${lrs3} --landmark ${lrs3}/landmark --manifest ${lrs3}/file.list \
 --cnn_detector /path/to/dlib_cnn_detector --face_predictor /path/to/dlib_landmark_predictor --ffmpeg /path/to/ffmpeg \
 --rank ${rank} --nshard ${nshard}
```
```sh
python align_mouth.py --video-direc ${lrs3} --landmark ${landmark_dir} --filename-path ${lrs3}/file.list \
 --save-direc ${lrs3}/video --mean-face /path/to/mean_face --s /path/to/ffmpeg \
 --rank ${rank} --nshard ${nshard}
```

This generates mouth ROIs in `${lrs3}/video`. It shards all videos in `${lrs3}/file.list` into `${nshard}` and generate mouth ROI for `${rank}`-th shard , where rank is an integer in `[0,nshard-1]`. The face detection and landmark prediction are done using [dlib](https://github.com/davisking/dlib). The links to download `cnn_detector`, `face_predictor`, `mean_face` can be found in the help message

### 3. Count number of frames per clip
```sh
python count_frames.py --root ${lrs3} --manifest ${lrs3}/file.list --nshard ${nshard} --rank ${rank}
```
This counts number of audio/video frames for `${rank}`-th shard and saves them in `${lrs3}/nframes.audio.${rank}` and `${lrs3}/nframes.video.${rank}` respectively. Merge shards by running:

```
for rank in $(seq 0 $((nshard - 1)));do cat ${lrs3}/nframes.audio.${rank}; done > ${lrs3}/nframes.audio
for rank in $(seq 0 $((nshard - 1)));do cat ${lrs3}/nframes.video.${rank}; done > ${lrs3}/nframes.video
```

If you are on slurm, the above commands (counting per shard + merging) can be combined by:
```sh
python count_frames_slurm.py --root ${lrs3} --manifest ${lrs3}/file.list --nshard ${nshard} \
 --slurm_partition ${slurm_partition}
```
It has dependency on [submitit](https://github.com/facebookincubator/submitit) and will directly  generate `${lrs3}/nframes.audio` and `${lrs3}/nframes.video`.

### 4. Set up data directory
```sh
python lrs3_manifest.py --lrs3 ${lrs3} --valid-ids /path/to/valid --vocab-size ${vocab_size}
```

This sets up data directory of trainval-only (~30h training data) and pretrain+trainval (~433h training data). It will first make a tokenizer based on sentencepiece model and set up target directory containing `${train|valid|test}.{tsv|wrd}`. `*.tsv` are manifest files and `*.wrd` are text labels.  `/path/to/valid` contains held-out clip ids used as validation set. The one used in our experiments can be found [here](data/lrs3-valid.id). 


## VoxCeleb2 Preprocessing

Download and decompress the [data](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html). Assume the data directory is `${vox}`, which contains two folders (`dev,test`). Follow the steps below:

### 1. Data preparation
```sh
python vox_prepare.py --vox ${vox} --ffmpeg /path/to/ffmpeg --rank ${rank} --nshard ${nshard} --step ${step}
```
This will generate a list of file-ids (`${vox}/file.list`, by step 1) and extract audio wavform from original videos (by step 2). `${step}` ranges from `1,2`.   `${nshard}` and `${rank}` are only used in step 2. This would shard all videos into `${nshard}` and extract audio for `${rank}`-th shard, where rank is an integer in `[0,nshard-1]`. 

### 2. Detect facial landmark and crop mouth ROIs:
```sh
python detect_landmark.py --root ${vox} --landmark ${vox}/landmark --manifest ${vox}/file.list \
 --cnn_detector /path/to/dlib_cnn_detector --face_predictor /path/to/dlib_landmark_predictor --ffmpeg /path/to/ffmpeg \
 --rank ${rank} --nshard ${nshard}
```
```sh
python align_mouth.py --video-direc ${vox} --landmark ${landmark_dir} --filename-path ${vox}/file.list \
 --save-direc ${vox}/video --mean-face /path/to/mean_face --ffmpeg /path/to/ffmpeg \
 --rank ${rank} --nshard ${nshard}
```

This generates mouth ROIs in `${vox}/video`, similar to LRS3 data preparation.

### 3. Count number of frames per clip
```sh
python count_frames.py --root ${vox} --manifest ${vox}/file.list --nshard ${nshard} --rank ${rank}
```
This counts number of audio/video frames for `${rank}`-th shard and saves them in `${vox}/nframes.audio.${rank}` and `${vox}/nframes.video.${rank}` respectively, similar to LRS3 data preparation. Merge shards by running:

```
for rank in $(seq 0 $((nshard - 1)));do cat ${vox}/nframes.audio.${rank}; done > ${vox}/nframes.audio
for rank in $(seq 0 $((nshard - 1)));do cat ${vox}/nframes.video.${rank}; done > ${vox}/nframes.video
```

If you are on slurm, the above commands (counting per shard + merging) can be combined by:
```sh
python count_frames_slurm.py --root ${vox} --manifest ${vox}/file.list --nshard ${nshard} \
 --slurm_partition ${slurm_partition}
```
It has dependency on [submitit](https://github.com/facebookincubator/submitit) and will directly  generate `${vox}/nframes.audio` and `${vox}/nframes.video`, similar to LRS3 data preparation.

### 4. Set up data directory
```sh
python vox_manifest.py --vox ${vox} --manifest ${vox}/file.list \
 --en-ids /path/to/en
```

This sets up data directory of the whole VoxCeleb2 and its English-only subset.  `/path/to/en` contains English-only clip ids. The one used in our experiments can be found [here](data/vox-en.id.gz). 

## Audio Noise Preparation (Optional)
If you want to test your model under noisy setting, you should prepare audio noise data. First download and decompress the [MUSAN](https://www.openslr.org/17/) corpus. Assume the data directory is `${musan}`, which contains the following folders `{music,speech,noise}`.

### 1. MUSAN data preparation
```sh
python musan_prepare_slurm.py --musan ${musan} --nshard ${nshard}  --slurm_partition ${slurm_partition}
```
This will: (1) split raw audios into 10-second clips, (2) generate babble noise from MUSAN speech audio, (3) count number of frames per clip. The whole data will be sharded into `${nshard}` parts and each job processes one part. It runs on Slurm and has dependency on [submitit](https://github.com/facebookincubator/submitit),

or
```sh
python musan_prepare.py --musan ${musan} --rank ${rank} --nshard ${nshard}
```

### 2. LRS3 audio noise preparation
```sh
python lrs3_noise.py --lrs3 ${lrs3}
```
It will generate LRS3 babble and speech noise including their manifest files, which are stored in `${lrs3}/noise/{babble,speech}`. `${lrs3}` is the LRS3 data directory. Make sure you already finished setting up LRS3 before running the command.

The following command generates babble noise from LRS3 training set.
```sh
python mix_babble.py --lrs3 ${lrs3}
```


### 3. Set up noise directory
```sh
python noise_manifest.py --lrs3 ${lrs3}  --musan ${musan}
```
It will make manifest (tsv) files for MUSAN babble, music and noise in `${musan}/tsv/{babble,music,noise}`, as well as a combined manifest in `${musan}/tsv/all` including MUSAN babble, music, noise and LRS3 speech. 


## DEMAND Noise Preparation (Optional)
Download the DEMAND noise dataset (16kHz), and place them like the below structure.
```
dataset/DEMAND/
├── DKITCHEN/
│  ├── ch01.wav
|  ├── ch02.wav
|  ├── ...
├── SCAFE/
│  ├── ch01.wav
|  ├── ch02.wav
|  ├── ...
```

Then, run this:
```sh
python demand_prepare.py --demand ${demand} --rank ${rank} --nshard ${nshard}
python demand_manifest.py --demand ${demand}
```

It only generates test.tsv files.



## Ego4D Preprocessing

Download by 
```sh
ego4d --output_directory="./ego4d_data" --datasets clips annotations
```

```sh
python ego4d_extract_transcripts_oracle.py ./ego4d_data/v2/annotations/av_val.json ${outdir}/ref.val.csv ${outdir}/ref.val.trn
python ego4d_extract_transcripts_oracle.py ./ego4d_data/v2/annotations/av_train.json ${outdir}/ref.train.csv ${outdir}/ref.train.trn
```
For train set, we eliminate the clips that durate over 15 sec or less than 0.4 sec.

Trimming videos and audios in short formats:
- `split`: `train` or `val`
```sh
python ego4d_prepare.py --ego4d ${ego4d_data} --output-dir ${outdir} --ffmpeg /usr/bin/ffmpeg --rank ${rank} --nshard ${nshard} --split ${split} --step 1
```

Build file and label list:
```sh
python ego4d_prepare.py --ego4d ${ego4d_data} --output-dir ${outdir} --ffmpeg /usr/bin/ffmpeg --rank ${rank} --nshard ${nshard} --split ${split} --step 2
```

Landmark detection (same as LRS3)
```sh
python detect_landmark.py --root ${outdir} --landmark ${outdir}/landmark --manifest ${outdir}/file.list \
  --cnn_detector /path/to/mmod_human_face_detector.dat --face_predictor /path/to/shape_predictor_68_face_landmarks.dat \
  --ffmpeg /path/to/ffmpeg --rank ${rank} --nshard 10
```
Recommended: use cuda-available dlib to speed up the detection ([link](https://gist.github.com/nguyenhoan1988/ed92d58054b985a1b45a521fcf8fa781))

```sh
python align_mouth.py --video-direc ${outdir} --landmark ${landmark_dir} --filename-path ${outdir}/file.list \
 --save-direc ${outdir}/video --mean-face /path/to/mean_face --ffmpeg /path/to/ffmpeg --rank ${rank} --nshard 10
```

```sh
python count_frames.py --root ${outdir} --manifest ${outdir}/file.list --nshard ${nshard} --rank ${rank}
```

```sh
python ego4d_manifest.py --ego4d ${outdir}
```