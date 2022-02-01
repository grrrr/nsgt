__version__ = '0.18'

from .cq import NSGT, CQ_NSGT
from .slicq import NSGT_sliced, CQ_NSGT_sliced
from .fscale import Scale, OctScale, LogScale, LinScale, MelScale, BarkScale, VQLogScale
from warnings import warn

try:
    from .audio import SndReader, SndWriter
except ImportError:
    warn("Audio IO routines (scikits.audio module) could not be imported")
