# CD1L validation notebooks

This installation is slightly different than with uv. We will work to bring them together. 

**Notes for install**: this currently installs the analysis tools including only GB waveforms. The MBH waveforms are in a private repo (not Michael's) that I am working on making public. In order to install the EMRI waveforms, you should uncomment the final block in the `notebooks/install.sh` file. This will install the proper branch of FEW. However, this can take awhile even with the proper Lapacke setup, so I have commented it out for now. 

**Note on Mojito package**: you will have to install the [Mojito package](https://mojito-e66317.io.esa.int). I had to request access and install from source from the gitlab to get it to work right. Otherwise, we may want to rewrite i/o operations on the cd1-L dataset to creat our own i/o functions (not the best option in my opinion). 

Steps for install:

1. Clone the cd1-L repo and change to notebooks directory:
```
git clone https://github.com/lisa-analysis-center/cd1l-validation.git
cd cd1l-validation/notebooks/
```

2. Generate a `virtualenv` (or something similar with conda).
```
python -m venv /path/to/cd1l_env
source activate /path/to/cd1l_env
```

3. If you have lapacke installed on your local machine, you should add it to cmake package config. For example, I have lapacke installed with brew:
```
export PKG_CONFIG_PATH="/opt/homebrew/opt/lapack/lib/pkgconfig:$PKG_CONFIG_PATH"
```
Otherwise, you should open `install.sh` and change `--config-settings=cmake.define.GBT_LAPACKE_DETECT_WITH=PKGCONFIG` to `--config-settings=cmake.define.GBT_LAPACKE_FETCH=ON` (multiple instances). This will fetch and install lapacke binaries as a part of the install process.

4. macOS only: If you have Homebrew's LLVM installed, make sure to use Apple's clang instead. Homebrew's clang++ uses a version of libc++ that is
incompatible with the system runtime dylib, causing symbol-not-found errors at import time. Force Apple clang with:
```
export CC=/usr/bin/clang
export CXX=/usr/bin/clang++
```
Alternatively, you can pass the compilers directly via pip without modifying your environment by adding `--config-settings=cmake.define.CMAKE_C_COMPILER=/usr/bin/clang --config-settings=cmake.define.CMAKE_CXX_COMPILER=/usr/bin/clang++` to each pip install call in install.sh.

5. Run the install script. The LISA specific codes currently operate on specific branches containing the TDI-on-the-fly, WDM, etc. updates. If you install with `install.sh`, it will install an editable environment where you can adjust the python codes in place. You can also compile and run c-codes in this infrastructure as well.
```
bash install.sh
```

5. Run the notebook:
```
jupyter notebook CD1-L-validation-gb.ipynb
```
