"""

    Python 2/3 Compatibility
    ~~~~~~~~~~~~~~~~~~~~~~~~

"""
try:
    xrange = xrange
except NameError:
    xrange = range

try:
    from itertools import izip, imap
except ImportError:
    izip = zip
    imap = map

try:
    from functools import reduce
except ImportError:
    reduce = reduce
