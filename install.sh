pip install --upgrade pip
pip install scikit_build_core uv uv_build setuptools_scm pybind11 numpy scipy ipython jupyter astropy lisaconstants Cython

#export CC=/usr/bin/clang
#export CXX=/usr/bin/clang++
#export PKG_CONFIG_PATH="/opt/homebrew/opt/lapack/lib/pkgconfig:$PKG_CONFIG_PATH"

# Eryn
git clone https://github.com/mikekatz04/Eryn.git
cd Eryn/
git checkout dev
pip install --no-build-isolation -e . --config-settings=cmake.define.GBT_LAPACKE_FETCH=ON
cd ../

# GPU Backend Tools
git clone https://github.com/mikekatz04/GPUBackendTools.git
cd GPUBackendTools/
git checkout spline
pip install --no-build-isolation -e . --config-settings=cmake.define.GBT_LAPACKE_FETCH=ON
cd ../

# LISA Analysis Tools
git clone https://github.com/mikekatz04/LISAanalysistools.git
cd LISAanalysistools/
git checkout dev
pip install --no-build-isolation -e . --config-settings=cmake.define.GBT_LAPACKE_FETCH=ON
cd ../

# fast lisa response
git clone https://github.com/mikekatz04/lisa-on-gpu.git
cd lisa-on-gpu/
git checkout tdi_on_fly
pip install --no-build-isolation -e . --config-settings=cmake.define.GBT_LAPACKE_FETCH=ON
cd ../

# phentax (MBH)
pip install git+https://github.com/asantini29/phentax.git

# Fast EMRI Waveforms
git clone https://github.com/BlackHolePerturbationToolkit/FastEMRIWaveforms.git
cd FastEMRIWaveforms/
git checkout gpu_backend
pip install --no-build-isolation -e . --config-settings=cmake.define.FEW_LAPACKE_DETECT_WITH=PKGCONFIG
cd ../
