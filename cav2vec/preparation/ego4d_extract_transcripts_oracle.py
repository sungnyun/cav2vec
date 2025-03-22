import sys, os
import json


def main(args):
    max_duration = 15
    min_interval = 0.4
    clip2trn = {}
    is_train = 'av_train' in args.subset_json

    with open(args.subset_json, 'r') as f:
        av_json = json.load(f)

        clip2segments = {}
        for video in av_json["videos"]:
            video_uid = video["video_uid"]
            for clip in video["clips"]:
                clip_uid = clip["clip_uid"]
                trn_list = []
                for trn in clip["transcriptions"]:
                    if trn["transcription"].strip() != "":
                        if is_train and \
                            (float(trn["end_time_sec"]) - float(trn["start_time_sec"]) > max_duration or \
                                float(trn["end_time_sec"]) - float(trn["start_time_sec"]) < min_interval):
                            continue
                        trn_list.append(
                            (
                                trn["start_time_sec"],
                                trn["end_time_sec"],
                                trn["person_id"],
                                trn["transcription"].strip()
                            )
                        )

                clip2segments[clip_uid] = trn_list

    with open(args.out_csv, 'w') as f:
        with open(args.out_trn, 'w') as trn_f:
            f.write("utt_id,clip,spkr,tb,te,trn\n")
            for clip in sorted(clip2segments):
                i_raw = -1
                for segm in clip2segments[clip]:
                    i_raw += 1
                    spkr = segm[2] if segm[2] != '' else "X"
                    tb, te =  segm[0], segm[1]
                    trn = segm[3]
                    utt_id = "{}-{}_1_{}_{}".format(spkr, clip, tb, te)
                    f.write("{},{},{},{},{},{}\n".format(
                        utt_id, clip + f"_{i_raw}", spkr, tb, te, trn))
                    trn_f.write("{} ({})\n".format(trn, utt_id))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "subset_json",
        type=str,
        help="Input JSON file that contains train/val/test set info"
    )
    parser.add_argument(
        "out_csv",
        type=str,
        help="Output CSV file that contains clip to transcription mapping"
    )
    parser.add_argument(
        "out_trn",
        type=str,
        help="Output TRN file that has transcriptions in SCLITE trn format"
    )

    args = parser.parse_args()
    main(args)