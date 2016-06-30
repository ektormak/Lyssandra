import ctypes
import yaml
import lyssa
import os

packagedir = lyssa.__path__[0]

def get_config():
    #get the configuration file
    #if not found return None
    try:
    	fidx = os.listdir(os.path.dirname(packagedir)).index("config.yml")
    	config_file = os.path.join(os.path.dirname(packagedir),os.listdir(os.path.dirname(packagedir))[fidx])
    	with open(config_file, 'r') as handle:
    		conf = yaml.load(handle)

    except:
    	#config file not found
    	conf = None
    return conf

conf = get_config()

def default_save_path():
    home = os.getenv("HOME")
    save_path = os.path.join(home,"lyssa_files")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    return save_path


if conf is not None:
    #set openblas
    if "openblas" in conf.keys():
        openblas_paths = [os.path.join(conf["openblas"],"libopenblas.so"),os.path.join(conf["openblas"],"libopenblas.so.0")]
        openblas_lib = None

        for libpath in openblas_paths:
        	try:
        		openblas_lib = ctypes.cdll.LoadLibrary(libpath)
        		has_openblas = True
        		break
        	except OSError:
        		continue
        if openblas_lib is None:
        	has_openblas = False
        	print "Could not locate an OpenBLAS shared library"
    else:
        has_openblas = False
    #set save paths
    if "paths" in conf.keys():
        save_paths = conf["paths"]
    else:
        save_paths = [default_save_path()]
else:
    #no config file found
    #using default configuration
    has_openblas = False
    save_paths = [default_save_path()]

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
	#every supplied path is invalid
	#use the default one
	save_paths = [default_save_path()]

max_workspaces = int(1e3)
