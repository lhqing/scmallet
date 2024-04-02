from importlib.metadata import version

__version__ = version("scmallet")

from .binarize import binarize_topics
from .mallet import Mallet
from .topic_metrices import (
    corr_array,
    corr_rows,
    dice_score_array,
    dice_score_rows,
)

# TODO
# generate pseudo-bulk and region bag
# write tests
# write documentation on README
