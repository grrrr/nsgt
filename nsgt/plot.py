import matplotlib.pyplot as plt
import torch
from nsgt.slicq import overlap_add_slicq


def spectrogram(c, fs, coef_factor, transform_name, freqs, frames, sliced=True, flatten=False, fontsize=14, cmap='inferno', slicq_name=''):
    # dB
    if not sliced:
        mls = 20.*torch.log10(torch.abs(c))
        transform_name = 'NSGT'
        print(f'mls shape nonsliced: {mls.shape}')
    else:
        chop = c.shape[-1]
        mls = 20.*torch.log10(torch.abs(overlap_add_slicq(c, flatten=flatten)))

        print(f'pre-remove slice portions from the sides, mls: {mls.shape}')
        mls = mls[:, :, :, int(chop/2):]
        mls = mls[:, :, :, :-int(chop/2)]
        print(f'remove slice portions from the sides, mls: {mls.shape}')

    plt.rcParams.update({'font.size': fontsize})
    fig, axs = plt.subplots(1)

    print(f"Plotting t*f space")

    # remove batch
    mls = torch.squeeze(mls, dim=0)
    # mix down multichannel
    mls = torch.mean(mls, dim=0)

    mls = mls.T

    fs_coef = fs*coef_factor # frame rate of coefficients
    mls_dur = len(mls)/fs_coef # final duration of MLS

    ncoefs = int(coef_factor*frames)

    print(f'ncoefs: {ncoefs}')
    mls = mls[:ncoefs, :]
    print(f'mls: {mls.shape}')

    mls_max = torch.quantile(mls, 0.999)
    im = axs.imshow(mls.T, interpolation='nearest', origin='lower', vmin=mls_max-120., vmax=mls_max, extent=(0,mls_dur,0,fs/2000), cmap=cmap, aspect=1000*mls_dur/fs)

    title = f'Magnitude {transform_name}'

    if slicq_name != '':
        title += f' {slicq_name}'

    axs.set_title(title)

    axs.set_xlabel('Time (s)')
    axs.set_ylabel('Frequency (kHz)')
    axs.yaxis.get_major_locator().set_params(integer=True)
    axs.set_yscale('linear')

    fig.colorbar(im, ax=axs, shrink=0.815, pad=0.006, label='dB')

    plt.subplots_adjust(wspace=0.001,hspace=0.001)
    plt.show()
