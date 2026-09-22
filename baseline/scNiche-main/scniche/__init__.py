import sys
from importlib.metadata import PackageNotFoundError, version

from . import datasets
from . import plot as pl
from . import preprocess as pp
from . import trainer as tr
from . import analysis as al

# has to be done at the end, after everything has been imported
sys.modules.update({f"{__name__}.{m}": globals()[m] for m in ["tr", "pp", "pl", "al"]})

try:
    __version__ = version("scniche")
except PackageNotFoundError:
    # 从源码目录加入 sys.path 但未 pip install -e . 时无分发元数据
    __version__ = "1.1.1+local"

__all__ = ["__version__", "datasets", "tr", "pp", "pl", "al"]

