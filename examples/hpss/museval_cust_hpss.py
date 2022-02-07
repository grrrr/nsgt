import os.path as op
import numpy as np
import os
import glob
import soundfile as sf
import functools
import warnings
import pandas as pd
from museval.aggregate import TrackStore, MethodStore, EvalStore, json2df
from museval import metrics, evaluate


def eval_mus_track_hpss(
    track_targets,
    user_estimates,
    track_name,
    track_rate,
    track_subset,
    output_dir=None,
    mode='v4',
    win=1.0,
    hop=1.0
):
    audio_estimates = []
    audio_reference = []

    # make sure to always build the list in the same order
    # therefore track.targets is an OrderedDict

    data = TrackStore(win=win, hop=hop, track_name=track_name)

    # compute evaluation of remaining targets
    for target in ['harmonic', 'percussive']:
        audio_estimates.append(user_estimates[target])
        audio_reference.append(track_targets[target])

    SDR, ISR, SIR, SAR = evaluate(
        audio_reference,
        audio_estimates,
        win=int(win*track_rate),
        hop=int(hop*track_rate),
        mode=mode
    )

    for i, target in enumerate(['harmonic', 'percussive']):
        values = {
            "SDR": SDR[i].tolist(),
            "SIR": SIR[i].tolist(),
            "ISR": ISR[i].tolist(),
            "SAR": SAR[i].tolist()
        }

        data.add_target(
            target_name=target,
            values=values
        )

    if output_dir:
        # validate against the schema
        data.validate()

        try:
            subset_path = op.join(
                output_dir,
                track_subset
            )

            if not op.exists(subset_path):
                os.makedirs(subset_path)

            with open(
                op.join(subset_path, track_name) + '.json', 'w+'
            ) as f:
                f.write(data.json)

        except (IOError):
            pass

    return data
