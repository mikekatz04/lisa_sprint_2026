# Global Fit Development

This replicates the development setup for NSGS pipeline development for the CD1-L (Mojito) data. The scripts included are a *small* \Delta away from the full global fit setup we will soon be running. 

It includes 3 scripts pe: `gb_test_script_td_wave.py`, `emri_test_script_td_wave.py`, `mbh_test_script_td_wave.py`.

These files include MCMC examples with the pipeline infrastructure for EMRIs, MBHs, and GBs (single GBs in this example). It include TDI-on-the-fly (from Neil and Tyson) and an implementation for the WDM domain. 

There is also the file `gb_lookup_table_test_script.py`. This file still needs some work and bug fixes that will hopefully be done by Monday. But it shows the lookup table method for generating GB waveforms (and hopefully eventually EMRI waveforms). 

**Notes for install**: this currently installs the LISA analysis tools development setup. The MBH waveforms are in a public repo from Alessandro Santini. In order to install the EMRI waveforms, you should uncomment the final block in the `install.sh` file. This will install the proper branch of FEW. However, this can take awhile even with the proper Lapacke setup, so I have commented it out for now. 

**GPUs**: This setup may run on GPUs. It may require a few bug fixes. If you try this, please let Michael Katz know how it goes. 

Steps for install:

1. Clone the cd1-L repo and change to notebooks directory:
```
git clone https://github.com/mikekatz04/lisa_sprint_2026.git
cd lisa_sprint_2026/
```

2. Generate a `virtualenv` (or something similar with conda).
```
python -m venv /path/to/sprint_env
source activate /path/to/sprint_env
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

5. Run the one of the scripts:
```
python mbh_test_script_td_wave.py
```
