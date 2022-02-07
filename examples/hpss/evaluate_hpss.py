import gc
import onair
import musdb
import os
import tqdm
import argparse
import torch
#from torch.profiler import profile, record_function, ProfilerActivity
from museval_cust_hpss import eval_mus_track_hpss
import numpy as np
import random
import time
import sys
from warnings import warn
from hpss import ihpss_stft, hpss_nsgt


def run_model(track, model, eval_dir=None, npy=False):
    N = track.audio.shape[0]

    track_audio = torch.unsqueeze(torch.tensor(
        track.audio.T,
        dtype=torch.float32,
        device="cpu",
        requires_grad=False
    ), dim=0)

    estimates = {}

    start = time.time()
    if not npy:
        target_estimates = torch.squeeze(model(track_audio), dim=0)
        end = time.time()

        # back to numpy
        estimates['harmonic'] = target_estimates[0, ...].detach().cpu().numpy().T + target_estimates[2, ...].detach().cpu().numpy().T + target_estimates[3, ...].detach().cpu().numpy().T
        estimates['percussive'] = target_estimates[1, ...].detach().cpu().numpy().T
    else:
        # apply model forward/inference with numpy
        target_estimates = model(torch.squeeze(track_audio.detach().cpu(), dim=0).numpy())
        end = time.time()

        estimates['harmonic'] = target_estimates[0, ...].detach().cpu().numpy().T
        estimates['percussive'] = target_estimates[1, ...].detach().cpu().numpy().T

    track_targets = {
        'harmonic': track.targets['vocals'].audio + track.targets['other'].audio + track.targets['bass'].audio,
        'percussive': track.targets['drums'].audio,
    }

    print(f'bss evaluation to store in {eval_dir}')
    bss_scores = eval_mus_track_hpss(
        track_targets,
        estimates,
        track.name,
        track.rate,
        track.subset,
        output_dir=eval_dir,
    )

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
        help='path to dataset root'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='musdb18hq',
        help='dataset (options: musdb18hq, onair)'
    )
    parser.add_argument(
        '--track-offset',
        type=int,
        default=0,
    )
    parser.add_argument(
        '--track-name',
        type=str,
        default='',
        help='track name (if empty, use all tracks)'
    )
    parser.add_argument(
        '--model-name',
        type=str,
        default='',
        help='model name (if empty, evaluate all models)'
    )

    args = parser.parse_args()

    # initiate OnAir music dataset
    if args.dataset == 'onair':
        mus = onair.DB(root=args.root)
    elif args.dataset == 'musdb18hq':
        mus = musdb.DB(root=args.root, subsets='test', is_wav=True)

    models = {
        'hpss-s12': lambda x: hpss_nsgt(x, 12, 'zeropad'),
        'hpss-s24': lambda x: hpss_nsgt(x, 24, 'zeropad'),
        'hpss-s48': lambda x: hpss_nsgt(x, 48, 'zeropad'),
        'hpss-s96': lambda x: hpss_nsgt(x, 96, 'zeropad'),
        'hpss-n12': lambda x: hpss_nsgt(x, 12, 'zeropad', nonsliced=True),
        'hpss-n24': lambda x: hpss_nsgt(x, 24, 'zeropad', nonsliced=True),
        'hpss-n48': lambda x: hpss_nsgt(x, 48, 'zeropad', nonsliced=True),
        'hpss-n96': lambda x: hpss_nsgt(x, 96, 'zeropad', nonsliced=True),
        'hpss-256': lambda x: ihpss_stft(x, harmonic_frame=256, harmonic_margin=1.0),
        'hpss-1024': lambda x: ihpss_stft(x, harmonic_frame=1024, harmonic_margin=1.0),
        'hpss-4096': lambda x: ihpss_stft(x, harmonic_frame=4096, harmonic_margin=1.0),
        'hpss-16384': lambda x: ihpss_stft(x, harmonic_frame=16384, harmonic_margin=1.0),
    }

    times = {k: 0. for k in models.keys()}

    pbar = tqdm.tqdm(mus.tracks[args.track_offset:])
    if len(pbar) == 0:
        raise ValueError('no tracks loaded, please use --root=/path/to/MUSDB18-HQ')

    for track in pbar:
        if args.track_name != '' and args.track_name != track.name:
            continue
        for model_name, model in models.items():
            if args.model_name != '' and args.model_name != model_name:
                continue
            print(f'evaluating track {track.name} with model {model_name}')
            est_path = os.path.join(args.eval_dir, f'{model_name}') if args.eval_dir else None
            aud_path = os.path.join(args.audio_dir, f'{model_name}') if args.audio_dir else None

            if 'hpss' in model_name:
                est, _, time_taken = run_model(track, model, eval_dir=est_path, npy=True)
            else:
                est, _, time_taken = run_model(track, model, eval_dir=est_path, npy=False)
            print(f'time {time_taken} s for song {track.name}')

            times[model_name] += time_taken

            gc.collect()

            if args.audio_dir:
                mus.save_estimates(est, track, aud_path)

    for model_name, model_time in times.items():
        print(f'{model_name}: time averaged per track: {model_time/len(pbar)}')
