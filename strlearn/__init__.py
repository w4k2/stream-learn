from ._version import __version__

from . import learners
from . import controllers
from . import ensembles
from . import utils
#from . import arff

__all__ = [
    'learners', 'controllers', 'ensembles', 'utils', 'arff', '__version__'
]
