import gc
import musdb
import os
import tqdm
import argparse
import torch
import museval
import numpy as np
import random
import time
import sys
from warnings import warn

from openunmix import umxhq


def run_model(track, model, eval_dir=None):
    N = track.audio.shape[0]

    track_audio = torch.unsqueeze(torch.tensor(
        track.audio.T,
        dtype=torch.float32,
        device="cpu"
    ), dim=0)

    start = time.time()

    # apply model forward/inference
    target_estimates = torch.squeeze(model(track_audio), dim=0)

    end = time.time()

    # assign to dict for eval
    estimates = {}
    accompaniment_source = 0

    for name, source in track.sources.items():
        # set this as the source estimate
        if name == 'vocals':
            estimates[name] = target_estimates[0, ...].detach().cpu().numpy().T
        elif name == 'bass':
            estimates[name] = target_estimates[bass_pos, ...].detach().cpu().numpy().T
        elif name == 'drums':
            estimates[name] = target_estimates[drums_pos, ...].detach().cpu().numpy().T
        elif name == 'other':
            estimates[name] = target_estimates[3, ...].detach().cpu().numpy().T

        # accumulate to the accompaniment if this is not vocals
        if name != 'vocals':
            accompaniment_source += estimates[name]

    estimates['accompaniment'] = accompaniment_source

    gc.collect()

    print(f'bss evaluation to store in {eval_dir}')
    bss_scores = museval.eval_mus_track(
        track,
        estimates,
        output_dir=eval_dir,
    )

    gc.collect()

    print(bss_scores)

    return estimates, bss_scores, end-start


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate HPSS vs. UMX'
    )
    parser.add_argument(
        '--audio_dir',
        nargs='?',
        help='Folder where audio results are saved',
        default=None,
    )
    parser.add_argument(
        '--eval_dir',
        nargs='?',
        help='Folder where evaluation results are saved'
    )
    parser.add_argument(
        '--root',
        type=str,
        default='',
        help='path to MUSDB18-HQ root'
    )

    args = parser.parse_args()

    # initiate musdb
    mus = musdb.DB(root=args.root, subsets='test', is_wav=True)

    # download umxhq (i.e. the one trained on MUSDB18-HQ)
    umx_separator = umxhq()

    models = {
        'umx': umx_separator,
    }

    times = {k: 0. for k in models.keys()}

    pbar = tqdm.tqdm(mus.tracks)
    if len(pbar) == 0:
        raise ValueError('no tracks loaded, please use --root=/path/to/MUSDB18-HQ')

    for track in pbar:
        for model_name, model in models.items():
            print(f'evaluating track {track.name} with model {model_name}')
            est_path = os.path.join(args.eval_dir, f'{model_name}') if args.eval_dir else None
            aud_path = os.path.join(args.audio_dir, f'{model_name}') if args.audio_dir else None

            est, _, time_taken = run_model(track, model, eval_dir=est_path)
            print(f'time {time_taken} s for song {track.name}')

            times[model_name] += time_taken

            gc.collect()

            if args.audio_dir:
                mus.save_estimates(est, track, aud_path)


    for model_name, model_time in times.items():
        print(f'{model_name}: time averaged per track: {model_time/len(pbar)}')
