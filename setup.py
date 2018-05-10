from setuptools import setup, find_packages
import os

datafiles = list((d, [os.path.join(d, f) for f in files]) for d, folders, files in os.walk("lyssa/utils/data_files"))

datafiles.append(("config.yml"))

setup(name="lyssa", version="0.0.4",
      author="Ektor Makridis",
      author_email="ektor.mak@gmail.com",
      data_files=datafiles,
      packages=find_packages())
