import ctypes
import yaml
import lyssa
import os

packagedir = lyssa.__path__[0]


def get_config():
    conf = None
    try:
        fidx = os.listdir(os.path.dirname(packagedir)).index("config.yml")
        config_file = os.path.join(os.path.dirname(packagedir), os.listdir(os.path.dirname(packagedir))[fidx])
        with open(config_file, 'r') as handle:
            conf = yaml.load(handle)
    except (IOError, OSError):
        pass
    return conf

conf = get_config()


def default_save_path():
    home = os.getenv("HOME")
    save_path = os.path.join(home, "lyssa_files")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    return save_path

save_paths = [default_save_path()]
openblas_lib = None

if conf is not None:
    # set openblas
    if "openblas" in conf.keys():
        openblas_paths = [os.path.join(conf["openblas"], "libopenblas.so"),
                          os.path.join(conf["openblas"], "libopenblas.so.0")]
        for libpath in openblas_paths:
            try:
                openblas_lib = ctypes.cdll.LoadLibrary(libpath)
                break
            except OSError:
                continue
        if not openblas_lib:
            print "could not locate an OpenBLAS shared library"
    else:
        has_openblas = False
    # set save paths
    if "paths" in conf.keys():
        save_paths = conf["paths"]


#find OpenBlas
#openblas_paths = [ '/opt/OpenBLAS/lib/libopenblas.so.0','/opt/OpenBLAS/lib/libopenblas.so',
#						'/opt/openblas/lib/libopenblas.so.0','/opt/openblas/lib/libopenblas.so']
n_invalid_paths = 0
for save_path in save_paths:
    if not os.path.exists(save_path):
        try:
            os.makedirs(save_path)
        except:
            print "could not create {0}".format(save_path)
            n_invalid_paths += 1

if n_invalid_paths == len(save_paths):
    save_path = default_save_path()
    print "using default save path {0}".format(save_path)
    save_paths = [save_path]

max_workspaces = int(1e3)
