mkdir /tmp/build
cd /tmp/build

# Build latest stable release from OpenBLAS from source
git clone -q --branch=master git://github.com/xianyi/OpenBLAS.git

(cd OpenBLAS && make FC=gfortran && make PREFIX=/opt/OpenBLAS install)

# Rebuild ld cache, this assumes that:
# /etc/ld.so.conf.d/openblas.conf was installed by Dockerfile
# and that the libraries are in /opt/OpenBLAS/lib
ldconfig

cd /
rm -rf /tmp/build
rm -rf /build_open_blas.sh
