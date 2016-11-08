FROM ubuntu:14.04
MAINTAINER Ektor Makridis <ektor.mak@gmail.com>

RUN apt-get update && apt-get install -y python-yaml python-PIL

ADD openblas.conf /etc/ld.so.conf.d/openblas.conf

#numpy, scipy config files
ADD numpy-site.cfg numpy-site.cfg
ADD scipy-site.cfg scipy-site.cfg

ADD build_open_blas.sh build_open_blas.sh
ADD build_numpy_scipy.sh build_numpy_scipy.sh

RUN bash build_open_blas.sh
RUN bash build_numpy_scipy.sh
RUN pip install git+git://github.com/scikit-learn/scikit-learn.git

RUN git clone https://github.com/ektormak/Lyssandra.git
