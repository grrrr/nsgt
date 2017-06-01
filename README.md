NSGT
====

Python implementation of Non-Stationary Gabor Transform (NSGT)
derived from MATLAB code by NUHAG, University of Vienna, Austria

Thomas Grill, 2011-2015
http://grrrr.org/nsgt

Austrian Research Institute for Artificial Intelligence (OFAI)
AudioMiner project, supported by Vienna Science and Technology Fund (WWTF)

covered by the Artistic License 2.0
http://www.perlfoundation.org/artistic_license_2_0


Mandatory dependencies:
-----------------------
Numerical Python (http://numpy.scipy.org)


Optional dependencies:
-----------------------
PyFFTW3 (https://launchpad.net/pyfftw)
will greatly speed up the NSGT transformation is fftw3 is installed on your system

scikits.audiolab (http://pypi.python.org/pypi/scikits.audiolab)
is needed for using the built-in audio import/streaming functionality


Installation:
-------------

Directly from GitHub:

``pip install git+git://github.com/grrrr/nsgt@master``

Note: if you are using Python3 on macOS or Linux, you may wish to replace `pip` with `pip3`.

ToDO:
-----

- Quality measurement for coefficients of sliced transform
- Unify nsgfwin sliced/non-sliced


Source:
-------

Original matlab code copyright follows:

AUTHOR(s) : Monika Dörfler, Gino Angelo Velasco, Nicki Holighaus, 2010-2011

COPYRIGHT : (c) NUHAG, Dept.Math., University of Vienna, AUSTRIA
http://nuhag.eu/
Permission is granted to modify and re-distribute this
code in any manner as long as this notice is preserved.
All standard disclaimers apply.
