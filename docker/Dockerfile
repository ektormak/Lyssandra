FROM ubuntu:14.04
MAINTAINER Ektor Makridis <ektor.mak@gmail.com>


#necessary files

#for ldconfig
ADD openblas.conf /etc/ld.so.conf.d/openblas.conf
#numpy,scipy config files
ADD numpy-site.cfg numpy-site.cfg
ADD scipy-site.cfg scipy-site.cfg
ADD build_open_blas.sh build_open_blas.sh
ADD build_numpy_scipy.sh build_numpy_scipy.sh


#install OpenBlas
RUN bash build_open_blas.sh
#install numpy,scipy
RUN bash build_numpy_scipy.sh
#install sklearn
RUN pip install git+git://github.com/scikit-learn/scikit-learn.git

RUN apt-get install -y python-yaml python-PIL
#libjpeg-dev libjpeg8-dev libpng3 libfreetype6-dev
RUN pip install boto

#make sure you pull the latest changes
RUN git clone https://ektormak:javabean12@github.com/ektormak/lyssa
#CMD cd lyssa && git pull && python setup.py build && python setup.py install

