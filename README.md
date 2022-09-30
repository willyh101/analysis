# New analysis code development
My personal repo for Adesnik Lab data analysis. General warning: Lots of monsters and untested code within!! Use at your own peril... Subject to code breaking changes at any time.


## Installation
* Use [mamba](https://mamba.readthedocs.io/en/latest/#) to manage the environment (you can use conda, it's just slower) (mamba easily installs on top of conda)
* Clone repo from GitHub
* Change to holofun directory
* Create environment with `mamba env create -f environment.yml`
* Install analysis package with `pip install  -e .`
* To install [OASIS](https://github.com/j-friedrich/OASIS), clone repo and build according to their instructions.
* To get it to install on windows install [C++ build tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/).
    * There might be [other ways to](https://gist.github.com/srikanthbojja/5c4205e37d28e1fb2b5b45e2e907b419) get it to install via conda listed compilers, but I have not tried that.

* `env2.yml` is the alternate build from conda-forge. These packages are tpyically more up to date. You will not get a MKL numpy build (might not matter for basic analysis unless you start doing lots of FFTs, matmul, dots, etc.). Be sure to comment out jax[cpu] if you are on windows (will not build).
  

#### Other useful conda/mamba tips
* To force a clean reinstall (recommended if you are making significant changes to more complicated libraries or want to install dask, numba, or jax)
  
  
  ```
  mamba env create --file environment.yml --force

  # alternatively, for update from the yml
  mamba env update --file environment.yml  --prune
  ```

  * Note that `--file environment.yml` can be left out of command if you are in the directory with that file (and it is named as such)


## Modules
* .tiffs    -> `SItiff`, a decently well tested class for loading ScanImage tiffs
* .daq      -> `SetupDaqFile`, a mostly working class for setupdaq data files
* .s2p      -> `Suite2pData`, a still in-development class for loading python s2p data
* .analysis -> generic tools for trace manipulation and processing