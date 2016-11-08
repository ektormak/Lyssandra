pip install cython

mkdir /tmp/build
cd /tmp/build

# Build NumPy and SciPy using OpenBLAS
git clone https://github.com/numpy/numpy
cp /numpy-site.cfg numpy/site.cfg
(cd numpy && python setup.py build && python setup.py install)

git clone https://github.com/scipy/scipy
cp /scipy-site.cfg scipy/site.cfg
(cd scipy && python setup.py build && python setup.py install)
# cleanup
cd /
rm -rf /tmp/build
rm -rf /build_numpy_scipy.sh
