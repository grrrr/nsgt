'''
copied from https://github.com/sigsep/sigsep-mus-2018-analysis/blob/master/aggregate.py
'''
from pathlib import Path
import pandas as pd
import json
import argparse


def museval2df(json_path):
    with open(json_path) as json_file:
        json_string = json.loads(json_file.read())
        #print(f"{json_string['targets']}")
        df = pd.json_normalize(
            json_string['targets'],
            ['frames'],
            ['name']
        )
        df = pd.melt(
            df,
            var_name='metric',
            value_name='score',
            id_vars=['time', 'name'],
            value_vars=['metrics.SDR', 'metrics.SAR', 'metrics.ISR', 'metrics.SIR']
        )
        df['track'] = json_path.stem
        df = df.rename(index=str, columns={"name": "target"})
        return df


def aggregate(input_dirs, output_path=None):
    data = []
    for path in input_dirs:
        p = Path(path)
        if p.exists():
            json_paths = p.glob('**/*.json')
            for json_path in json_paths:
                df = museval2df(json_path)
                df['method'] = p.stem
                print(df['method'])
                data.append(df)

    df = pd.concat(data, ignore_index=True)
    if output_path is not None:
        df.to_pickle(output_path)
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Aggregate Folder')
    parser.add_argument(
        'submission_dirs',
        help='directories of submissions',
        nargs='+',
        type=str
    )

    parser.add_argument(
        '--out',
        help='saves dataframe to disk',
        type=str
    )

    args = parser.parse_args()
    df = aggregate(args.submission_dirs, output_path=args.out)
