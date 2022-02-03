from .nsgfwin_sl import nsgfwin
from .nsdual import nsdual
from .nsgtf import nsgtf
from .nsigtf import nsigtf
from .util import calcwinrange
from .fscale import OctScale
from math import ceil
import torch


class NSGT(torch.nn.Module):
    def __init__(self, scale, fs, Ls, real=True, matrixform=False, reducedform=0, multichannel=False, dtype=torch.float32, device="cpu"):
        assert fs > 0
        assert Ls > 0
        assert 0 <= reducedform <= 2

        super(NSGT, self).__init__()

        self.scale = scale
        self.fs = fs
        self.Ls = Ls
        self.real = real
        self.reducedform = reducedform
        self.matrixform = matrixform

        self.device = torch.device(device)
        
        self.frqs,self.q = scale()

        # calculate transform parameters
        self.g,rfbas,self.M = nsgfwin(self.frqs, self.q, self.fs, self.Ls, sliced=False, dtype=dtype, device=self.device)

        if real:
            assert 0 <= reducedform <= 2
            sl = slice(reducedform,len(self.g)//2+1-reducedform)
        else:
            sl = slice(0,None)

        self.fbins_actual = sl.stop

        # coefficients per slice
        self.ncoefs = max(int(ceil(float(len(gii))/mii))*mii for mii,gii in zip(self.M[sl],self.g[sl]))        

        if matrixform:
            if self.reducedform:
                rm = self.M[self.reducedform:len(self.M)//2+1-self.reducedform]
                self.M[:] = rm.max()
            else:
                self.M[:] = self.M.max()
    
        if multichannel:
            self.channelize = lambda s: s
            self.unchannelize = lambda s: s
        else:
            self.channelize = lambda s: (s,)
            self.unchannelize = lambda s: s[0]

        # calculate shifts
        self.wins,self.nn = calcwinrange(self.g, rfbas, self.Ls, device=self.device)
        # calculate dual windows
        self.gd = nsdual(self.g, self.wins, self.nn, self.M, device=self.device)
        self.setup_lambdas()

    def setup_lambdas(self):
        self.fwd = lambda s: nsgtf(s, self.g, self.wins, self.nn, self.M, real=self.real, reducedform=self.reducedform, device=self.device, matrixform=self.matrixform)
        self.bwd = lambda c: nsigtf(c, self.gd, self.wins, self.nn, self.Ls, real=self.real, reducedform=self.reducedform, device=self.device)

    def _apply(self, fn):
        super(NSGT, self)._apply(fn)
        self.wins = [fn(w) for w in self.wins]
        self.g = [fn(g) for g in self.g]
        self.device = self.g[0].device
        self.setup_lambdas()

    @property
    def coef_factor(self):
        return float(self.ncoefs)/self.Ls
    
    @property
    def slice_coefs(self):
        return self.ncoefs
    
    def forward(self, s):
        'transform'
        s = self.channelize(s)
        #c = list(map(self.fwd, s))
        c = self.fwd(s)
        return self.unchannelize(c)

    def backward(self, c):
        'inverse transform'
        c = self.channelize(c)
        #s = list(map(self.bwd,c))
        s = self.bwd(c)
        return self.unchannelize(s)
    
class CQ_NSGT(NSGT):
    def __init__(self, fmin, fmax, bins, fs, Ls, real=True, matrixform=False, reducedform=0, multichannel=False):
        assert fmin > 0
        assert fmax > fmin
        assert bins > 0
        
        self.fmin = fmin
        self.fmax = fmax
        self.bins = bins

        scale = OctScale(fmin, fmax, bins)
        NSGT.__init__(self, scale, fs, Ls, real, matrixform=matrixform, reducedform=reducedform, multichannel=multichannel)


def make_nsgt_filterbanks(nsgt_base, sample_rate=44100.0):
    if sample_rate != 44100.0:
        raise ValueError('i was lazy and harcoded a lot of 44100.0, forgive me')

    encoder = TorchNSGT(nsgt_base)
    decoder = TorchINSGT(nsgt_base)

    return encoder, decoder


class NSGTBase(nn.Module):
    allowed_chunk_windows = ['rectangular', 'hann']

    def __init__(self,
        scale, fbins, fmin, N, fmax=22050, gamma=25.,
        matrixform='ragged',
        chunk_window='rectangular', n_overlap=0,
        fs=44100, device="cpu"
    ):
        super(NSGTBase, self).__init__()
        self.fbins = fbins
        self.fmin = fmin
        self.gamma = gamma
        self.fmax = fmax

        scl_fn = None
        self.scl = None
        scl_args = None
        try:
            scl_fn = SCALES_BY_NAME[scale]
        except KeyError:
            msg = f'unsupported frequency scale {scale}'
            if scale == 'oct':
                msg += '\n\tuse `cqlog` instead of `oct`'
            raise ValueError(msg)

        if scale == 'vqlog':
            scl_args = (self.fmin, self.fmax, self.fbins, self.gamma)
        else:
            scl_args = (self.fmin, self.fmax, self.fbins)
        self.scl = scl_fn(*scl_args)

        self.device = device
        self.fs = fs

        if matrixform not in ALLOWED_MATRIXFORMS:
            raise ValueError(f'{matrixform} is not one of the allowed values: {ALLOWED_MATRIXFORMS}')
        self.matrixform = matrixform

        if chunk_window not in NSGTBase.allowed_chunk_windows:
            raise ValueError(f"{chunk_window} is not one of the allowed values: {NSGTBase.allowed_chunk_windows}")
        self.chunk_window = chunk_window
        self.chunk_nwin = N
        self.chunk_overlap = chunk_overlap

        self.nsgt = None
        if self.matrixform == 'zeropad':
            self.nsgt = NSGT(self.scl, fs, N, real=True, multichannel=True, matrixform=True, device=self.device)
        else:
            self.nsgt = NSGT(self.scl, fs, N, real=True, multichannel=True, matrixform=False, device=self.device)

        self.M = self.nsgt.ncoefs
        self.fbins_actual = self.nsgt.fbins_actual

    def max_bins(self, bandwidth): # convert hz bandwidth into bins
        if bandwidth is None:
            return None
        freqs, _ = self.scl()
        max_bin = min(np.argwhere(freqs > bandwidth))[0]
        return max_bin+1

    def _apply(self, fn):
        self.nsgt._apply(fn)
        return self


class TorchNSGT(nn.Module):
    def __init__(self, nsgt):
        super(TorchNSGT, self).__init__()
        self.nsgt = nsgt

    def _apply(self, fn):
        self.nsgt._apply(fn)
        return self

    def forward(self, x):
        shape = x.size()
        nb_samples, nb_channels, nb_timesteps = shape

        # pack batch
        x = x.view(-1, shape[-1])

        C = self.nsgt.nsgt.forward(x)

        for i, nsgt_f in enumerate(C):
            nsgt_f = torch.view_as_real(nsgt_f)
            # unpack batch
            nsgt_f = nsgt_f.view(shape[:-1] + nsgt_f.shape[-3:])
            C[i] = nsgt_f

        if self.nsgt.matrixform == 'ragged':
            return C
        elif self.nsgt.matrixform == 'zeropad':
            return C[0]

        # interpolation here
        return C

    @torch.no_grad()
    def interpolate(self, nsgt: List[Tensor]) -> Tensor:
        nb_samples, nb_channels = slicq[0].shape[:2]
        total_f_bins = sum([slicq_.shape[-2] for slicq_ in slicq])
        max_t_bins = max([slicq_.shape[-1] for slicq_ in slicq])

        interpolated = torch.zeros((nb_samples, nb_channels, total_f_bins, max_t_bins), dtype=slicq[0].dtype, device=slicq[0].device)

        fbin_ptr = 0
        for i, slicq_ in enumerate(slicq):
            nb_samples, nb_channels, nb_f_bins, nb_t_bins = slicq_.shape

            if nb_t_bins == max_t_bins:
                # same time width, no interpolation
                interpolated[:, :, fbin_ptr:fbin_ptr+nb_f_bins, :] = slicq_
            else:
                # repeated interpolation
                interp_factor = max_t_bins//nb_t_bins
                max_assigned = nb_t_bins*interp_factor
                rem = max_t_bins - max_assigned
                interpolated[:, :, fbin_ptr:fbin_ptr+nb_f_bins, : max_assigned] = torch.repeat_interleave(slicq_, interp_factor, dim=-1)
                interpolated[:, :, fbin_ptr:fbin_ptr+nb_f_bins, max_assigned : ] = torch.unsqueeze(slicq_[..., -1], dim=-1).repeat(1, 1, 1, rem)
            fbin_ptr += nb_f_bins
        return interpolated

    @torch.no_grad()
    def deinterpolate(self, interpolated: Tensor, ragged_shapes: List[Tuple[int]]) -> List[Tensor]:
        max_t_bins = interpolated.shape[-1]
        full_slicq = []
        fbin_ptr = 0
        for i, bucket_shape in enumerate(ragged_shapes):
            curr_slicq = torch.zeros(bucket_shape, dtype=interpolated.dtype, device=interpolated.device)

            nb_t_bins = bucket_shape[-1]
            freqs = bucket_shape[-2]

            if bucket_shape[-1] == interpolated.shape[-1]:
                # same time width, no interpolation
                curr_slicq = interpolated[:, :, fbin_ptr:fbin_ptr+freqs, :]
            else:
                # inverse of repeated interpolation
                interp_factor = max_t_bins//nb_t_bins
                select = torch.arange(0, max_t_bins,interp_factor, device=interpolated.device)
                curr_slicq = torch.index_select(interpolated[:, :, fbin_ptr:fbin_ptr+freqs, :], -1, select)

            # crop just in case
            full_slicq.append(curr_slicq[..., : bucket_shape[-1]])

            fbin_ptr += freqs
        return full_slicq


class TorchINSGT(nn.Module):
    def __init__(self, nsgt):
        super(TorchINSGT, self).__init__()
        self.nsgt = nsgt

    def _apply(self, fn):
        self.nsgt._apply(fn)
        return self

    def forward(self, X_list, length: int) -> Tensor:
        X_complex = [None]*len(X_list)
        for i, X in enumerate(X_list):
            Xshape = len(X.shape)

            X = torch.view_as_complex(X)

            shape = X.shape

            if Xshape == 6:
                X = X.view(X.shape[0]*X.shape[1], *X.shape[2:])
            else:
                X = X.view(X.shape[0]*X.shape[1]*X.shape[2], *X.shape[3:])

            # moveaxis back into into T x [packed-channels] x F1 x F2
            X = torch.moveaxis(X, -2, 0)

            X_complex[i] = X

        y = self.nsgt.nsgt.backward(X_complex, length)

        # unpack batch
        y = y.view(*shape[:-3], -1)

        return y
