import sys
import os
import os.path

try:
    # py2.x
    from urllib import pathname2url
    pass
except ImportError:
    # py3.x
    from urllib.request import pathname2url
    pass


from .paths import __version__
from .paths import get_rscripts_path
from .paths import getstepurlpath

#from . import training_eval
