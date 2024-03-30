from importlib.metadata import version

__version__ = version("scmallet")

from .binarize import binarize_topics
from .mallet import Mallet

# TODO
# infer topics and add to adata
# combine or select redudant topics when given multiple topic models
# generate pseudo-bulk and region bag
