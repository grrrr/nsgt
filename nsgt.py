import torch


def _slicq_wins(window):
    ws = torch.ones(window)
    wa = torch.ones(window)

    return ws, wa


def overlap_add_slicq(slicq):
    nb_samples, nb_channels, nb_f_bins, nb_slices, nb_m_bins = slicq.shape

    window = nb_m_bins
    hop = window//2 # 50% overlap window

    ncoefs = nb_slices*nb_m_bins//2 + hop
    out = torch.zeros((nb_samples, nb_channels, nb_f_bins, ncoefs), dtype=slicq.dtype, device=slicq.device)

    w, _ = _slicq_wins(window)

    ptr = 0

    for i in range(nb_slices):
        out[:, :, :, ptr:ptr+window] += w*slicq[:, :, :, i, :]
        ptr += hop

    return out


def inverse_ola_slicq(slicq, nb_slices, nb_m_bins):
    nb_samples, nb_channels, nb_f_bins, ncoefs = slicq.shape

    window = nb_m_bins
    hop = window//2 # 50% overlap window

    assert(ncoefs == (nb_slices*nb_m_bins//2 + hop))

    out = torch.zeros((nb_samples, nb_channels, nb_f_bins, nb_slices, nb_m_bins), dtype=slicq.dtype, device=slicq.device)

    ptr = 0

    ws, wa = _slicq_wins(window)

    for i in range(nb_slices):
        out[:, :, :, i, :] += ws*slicq[:, :, :, ptr:ptr+window]
        ptr += hop

    out *= hop/torch.sum(ws*wa)
    return out


if __name__ == '__main__':
    nb_samples = 1
    nb_channels = 2
    nb_f_bins = 117
    nb_slices = 67
    nb_m_bins = 244

    slicq = torch.rand(size=(nb_samples, nb_channels, nb_f_bins, nb_slices, nb_m_bins))

    ola = overlap_add_slicq(slicq)

    slicq_inv = inverse_ola_slicq(ola, nb_slices, nb_m_bins)

    ola2 = overlap_add_slicq(slicq_inv)

    mse1 = torch.nn.functional.mse_loss(slicq, slicq_inv)
    mse2 = torch.nn.functional.mse_loss(ola, ola2)

    print(f'{torch.allclose(slicq, slicq_inv)}')
    print(f'{mse1} {mse2}')
