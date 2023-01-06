import logging
# custom logging
logformat = '{relativeCreated:08.0f} - {levelname:8} - [{module}:{funcName}:{lineno}] - {message}'
logging.basicConfig(format=logformat, style='{')

# scope some useful things locally
from .utils import (
    make_paths,
    nbsetup,
    make_results_folder
)