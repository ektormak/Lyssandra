FROM ubuntu:14.04
MAINTAINER Ektor Makridis <ektor.mak@gmail.com>

RUN apt-get update && \
    apt-get install -y \
    python-yaml \
    python-PIL \
    build-dep \
    python-imaging
    libjpeg8
    libjpeg62-dev
    libfreetype6
    libfreetype6-dev
    build-essential
    python-dev
    python-setuptools
    python-pip gfortran git


ADD openblas.conf /etc/ld.so.conf.d/openblas.conf
#numpy, scipy config files
ADD numpy-site.cfg numpy-site.cfg
ADD scipy-site.cfg scipy-site.cfg

ADD build_open_blas.sh build_open_blas.sh
ADD build_numpy_scipy.sh build_numpy_scipy.sh

RUN bash build_open_blas.sh
RUN bash build_numpy_scipy.sh

RUN pip install -U pip && \
    pip install Pillow
    pip install git+git://github.com/scikit-learn/scikit-learn.git

RUN git clone https://github.com/ektormak/Lyssandra.git
