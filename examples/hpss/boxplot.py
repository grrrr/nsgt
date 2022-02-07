'''
adapted from https://github.com/sigsep/sigsep-mus-2018-analysis/blob/master/sisec-2018-paper-figures/boxplot.py
'''

import pandas
import seaborn
import argparse
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib
import sys
import gc
import itertools

#controls = ['irm1-s1024', 'irm1-s4096', 'irm1-s16384'] 
#controls = ['mpi-4096', 'irm1-4096']
#controls = []
controls = ['hpss-1024']


def save_boxplot(pandas_in, pdf_out, single=False, colors_legend=None, print_median_only=False):
    metrics = ['SDR', 'SIR', 'SAR', 'ISR']
    targets = ['harmonic', 'percussive']

    df = pandas.read_pickle(pandas_in)
    df['control'] = df.method.isin(controls)

    df.replace('metrics.SDR', value='SDR', inplace=True)
    df.replace('metrics.SAR', value='SAR', inplace=True)
    df.replace('metrics.SIR', value='SIR', inplace=True)
    df.replace('metrics.ISR', value='ISR', inplace=True)

    # aggregate methods by mean using median by track
    df = df.groupby(
        ['method', 'track', 'target', 'metric']
    ).median().reset_index()

    pandas.set_option('display.max_colwidth', None)
    pandas.set_option('display.max_columns', None)  
    pandas.set_option('display.max_rows', None)  

    #print(df[(df.metric == "SDR")].groupby(
    #    ['method', 'target', 'metric']
    #).median('time'))

    print('median sdr:\n{0}'.format(
            df[(df.metric == "SDR") & (df.target != 'accompaniment')].groupby(
            ['method']
            ).median('time')))

    if print_median_only:
        sys.exit(0)

    # Get sorting keys (sorted by median of SDR:vocals)
    df_sort_by = df[
        (df.metric == "SDR") &
        (df.target == "harmonic")
    ]

    methods_by_sdr = df_sort_by.score.groupby(
        df_sort_by.method
    ).median().sort_values().index.tolist()

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    seaborn.set()
    seaborn.set_context("paper")

    params = {
        'backend': 'ps',
        'axes.labelsize': 18,
        'font.size': 15,
        'legend.fontsize': 16,
        'xtick.labelsize': 13,
        'ytick.labelsize': 15,
        'text.usetex': True,
        'font.family': 'serif',
        'font.serif': 'ptmrr8re',
    }

    seaborn.set_style("darkgrid", {
        'pgf.texsystem': 'xelatex',  # pdflatex, xelatex, lualatex
        "axes.facecolor": "0.925",
        'text.usetex': True,
        'font.family': 'serif',
        'axes.labelsize': 14,
        'font.size': 14,
        'legend.fontsize': 15,
        'xtick.labelsize': 15,
        'ytick.labelsize': 17,
        'font.serif': [],
    })
    plt.rcParams.update(params)

    if single:
        with PdfPages(pdf_out) as pdf:
            for (target, metric) in itertools.product(targets, metrics):
                g = seaborn.boxplot(
                    x="score",
                    y="method",
                    hue="control",
                    data=df.loc[(df['target'] == target) & (df['metric'] == metric)],
                    orient='h',
                    order=methods_by_sdr[::-1],
                    hue_order=[False, True],
                    showfliers=False,
                    notch=True,
                )
                g.legend_.remove()
                g.figure.suptitle(f'{target} - {metric}')
                g.figure.set_size_inches(8.5,11)
                g.figure.tight_layout()
                g.figure.savefig(
                    pdf,
                    format='pdf',
                    bbox_inches='tight',
                    dpi=300,
                )
                del g
                gc.collect()
                plt.clf()
    else:
        g = seaborn.FacetGrid(
            df,
            row="target",
            col="metric",
            row_order=targets,
            col_order=metrics,
            height=6,
            sharex=False,
            aspect=0.7
        )
        g = (g.map(
            seaborn.boxplot,
            "score",
            "method",
            "control",
            orient='h',
            order=methods_by_sdr[::-1],
            hue_order=[False, True],
            showfliers=False,
            notch=True
        ))

        if colors_legend is not None:
            if colors_legend == 'control':
                for (row_val, col_val), ax in g.axes_dict.items():
                    ax.artists[0].set_facecolor('magenta')
                    #ax.artists[1].set_facecolor('magenta')
                    #ax.artists[2].set_facecolor('orangered')
                    #ax.artists[3].set_facecolor('orangered')
                    ax.artists[2].set_facecolor('cyan')
                    #ax.artists[6].set_facecolor('cyan')

                name_to_color = {
                    'chosen-slicqt': 'magenta',
                    #'bad-slicq': 'orangered',
                    'control-stft': 'cyan',
                }

                patches = [matplotlib.patches.Patch(color=v, label=k) for k,v in name_to_color.items()]
                matplotlib.pyplot.legend(handles=patches, loc='center', bbox_to_anchor=(-1.5, -0.25), ncol=3)
            elif colors_legend == 'pretrained':
                for (row_val, col_val), ax in g.axes_dict.items():
                    #ax.artists[0].set_facecolor('magenta')
                    #ax.artists[1].set_facecolor('magenta')
                    #ax.artists[2].set_facecolor('cyan')
                    #ax.artists[3].set_facecolor('cyan')
                    ax.artists[6].set_facecolor('gold')
                    ax.artists[7].set_facecolor('darkviolet')

                name_to_color = {
                    'slicq-wslicq': 'gold',
                    'slicq-wstft': 'darkviolet',
                }

                patches = [matplotlib.patches.Patch(color=v, label=k) for k,v in name_to_color.items()]
                matplotlib.pyplot.legend(handles=patches, loc='center', bbox_to_anchor=(-1.5, -0.25), ncol=3)

        plt.setp(g.fig.texts, text="")
        g.set_titles(col_template="{col_name}", row_template="{row_name}")

        g.fig.tight_layout()
        plt.subplots_adjust(hspace=0.2, wspace=0.3)
        #plt.subplots_adjust(hspace=0.15, wspace=0.05)
        g.fig.set_size_inches(12,18)
        g.fig.savefig(
            pdf_out,
            bbox_inches='tight',
            dpi=400
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate boxplot')
    parser.add_argument(
        'pandas_in',
        type=str,
        help='in .pandas file generated by aggregate.py',
    )
    parser.add_argument(
        'pdf_out',
        type=str,
        help='path to output pdf file',
    )
    parser.add_argument(
        '--single',
        action='store_true',
        help='single boxplot per page'
    )
    parser.add_argument(
        '--print-median-only',
        action='store_true',
        help='only print the median sdr'
    )
    parser.add_argument(
        '--colors-legend',
        type=str,
        default=None,
        help='color legend (control vs. full eval)',
    )

    args = parser.parse_args()
    save_boxplot(args.pandas_in, args.pdf_out, args.single, colors_legend=args.colors_legend, print_median_only=args.print_median_only)
