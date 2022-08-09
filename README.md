# New analysis code development
My personal repo for Adesnik Lab data analysis. General warning: Lots of monsters and untested code within!! Use at your own peril... Subject to code breaking changes at any time.

## Installation
* Use [mamba](https://mamba.readthedocs.io/en/latest/#) to manage the environment
* Clone repo from GitHub
* Change to analysis directory
* Create environment with `mamba env create -f env.yml`
* Install analysis package with `pip install  -e .`
* To install [OASIS](https://github.com/j-friedrich/OASIS), clone repo and build according to their instructions.

## Useful things
* .tiffs -> `SItiff`, a decently well tested class for loading ScanImage tiffs
* .daq   -> `SetupDaqFile`, a mostly working class for setupdaq data files
* .s2p   -> `Suite2pData`, a still in-development class for loading python s2p data