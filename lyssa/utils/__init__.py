from .math import fast_dot,outer,norm,norm_cols
import numpy as np
import sys,os
import pickle
import warnings
from PIL import Image
from .config import has_openblas,openblas_lib
from .dataset import split_dataset,get_mmap,get_empty_mmap,img_dataset
from .workspace import max_mmap_files,mmap_base,workspace_manager, \
						show_workspaces,copy_workspace,get_workspace,request_workspace

import lyssa
from time import time

packagedir = lyssa.__path__[0]

def set_openblas_threads(n):
	"""
	Set the current number of threads used by the OpenBLAS server
	"""
	if has_openblas:
		openblas_lib.openblas_set_num_threads(int(n))

def get_openblas_threads():
	"""Get the current number of threads used by the OpenBLAS server
	"""
	return openblas_lib.openblas_get_num_threads() if has_openblas else 0

import multiprocessing
cpu_count = multiprocessing.cpu_count()

def touch(fname):
	open(fname, 'wa').close()

def run_parallel(func=None,data=None,args=None,batched_args=None,result_shape=None,batch_size=None,n_batches=None,mmap=False,msg=None,n_jobs=None):

	"""
	a high-order function that parallelizes an input function
	over its input arguments.

	convention:
	the first argument of func should be the data,
	next follow the batched arguments,
	finally the fixed arguments
	"""

	is_array = type(data) is np.ndarray or type(data) is np.core.memmap
	is_list = hasattr(data,'__len__')
	if is_array:
		n_samples = data.shape[1]
	elif is_list:
		n_samples = len(data)

	if args is None:
		args = ()

	if result_shape is not None:
		if type(result_shape) is tuple:
			if mmap:
				Z = get_empty_mmap(result_shape)
			else:
				Z = np.zeros(result_shape)
			result_is_array =  True
		else:
			#result shape is an integer which
			#implies that result will be a list
			Z = np.zeros(result_shape)
			result_is_array =  False
	else:
		Z = None

	if n_jobs == 1:

		_args = [data]
		if batched_args is not None:
			for i in range(len(batched_args)):
				_args.append(batched_args[i])
		for i in range(len(args)):
			_args.append(args[i])

		rs = func(*_args)
		if rs is not None:
			Z[:] = rs
		return Z

	pool = multiprocessing.Pool(processes=n_jobs,initializer=None)
	if n_batches is not None:
		idx = gen_even_batches(n_samples,n_batches)
	elif batch_size is not None:
		idx = gen_batches(n_samples,batch_size=batch_size)
		n_batches = len(idx)

	if batched_args is not None:
		n_batched_args = len(batched_args)
	else:
		n_batched_args = 0

	results = []
	for i in xrange(n_batches):

		if is_array:
			data_batch = data[:,idx[i]]
		elif is_list:
			start = idx[i][0]
			end = idx[i][-1]+1
			data_batch = data[start:end]


		#assume that batched args
		#are not keyword arguments
		_args = [data_batch]
		for j in range(n_batched_args):
			batched_arg = batched_args[j]
			batch_is_array = type(batched_arg) is np.ndarray or type(batched_arg) is np.core.memmap
			batch_is_list = hasattr(batched_arg,'__len__')
			if batch_is_array:
				batched_arg = batched_arg[:,idx[i]]
			elif batch_is_list:
				start = idx[i][0]
				end = idx[i][-1]+1
				batched_arg = batched_arg[start:end]
			_args.append(batched_arg)

		_args += list(args)
		results.append( (i,pool.apply_async(func, _args)) )

	time_per_task = 0
	n_tasks = n_batches
	t_start = time()
	eta_string = ""
	for i,result in results:
		if msg is not None:
			sys.stdout.write("\r"+msg+"...:%3.2f%%" % (( i / float(n_batches))*100)+eta_string)
			sys.stdout.flush()
		rs = result.get()
		if rs is not None:

			if rs.shape != Z[:,idx[i]].shape:
				import pdb; pdb.set_trace()
			if result_is_array:
				Z[:,idx[i]] = rs
			else:
				Z[idx[i]] = rs
			results[i] = None

		if i > 0:
			duration = time() - t_start
			time_per_task = duration / float(i)
			incomplete_count = n_tasks - i
			eta = incomplete_count * time_per_task
			mins, secs = divmod(eta, 60)
			hours, mins = divmod(mins, 60)
			#"%d:%02d:%02d" % (h, m, s)
			eta_string = " ,eta=%02d:%02d:%02d" % (hours,mins,secs)

	pool.close()
	if msg is not None:
		sys.stdout.write("\r"+msg+"...:%3.2f%%" % (100))
		sys.stdout.flush()
		print ""
	return Z


def gen_even_batches(N,n_batches):

	"""
	returns a list where each element is a list of indices
	of all the datapoint in the same batch
	"""
	base = 0
	batch_size =  int(np.floor(N / float(n_batches)))
	batches_idx = []
	for j in range(n_batches-1):
		batches_idx.append(range(base,base+batch_size))
		base += batch_size

	batches_idx.append(range(base,N))
	return batches_idx


def gen_batches(N,batch_size=None):

	"""
	same as gen_even_batches but it requires a fixed batch_size
	"""

	if batch_size is not None:

		n_batches =  int(np.floor(N / float(batch_size)))
		batches_idx = []
		base = 0
		for j in range(n_batches):
			batches_idx.append(range(base,base+batch_size))
			base += batch_size

		if N > base:
			batches_idx.append(range(base,N))
	return batches_idx



def get_data(filename):
	dirname = os.path.join(os.path.dirname(packagedir), 'lyssa','utils', 'data_files')
	fullname = os.path.join(dirname, filename)
	return fullname


def lena():
	img = Image.open( get_data('lena.png') )
	return np.asarray(img)


def girl():
	img = Image.open( get_data('girl.png') )
	return np.asarray(img)

def flowers():
	img = Image.open( get_data('flowers.png') )
	return np.asarray(img)

def alley():
	img = Image.open( get_data('alley.png') )
	return np.asarray(img)


def barbara():
	img = Image.open( get_data('barbara.png') )
	return np.asarray(img)


def man():
	img = Image.open( get_data('man.png') )
	return np.asarray(img)

def boat():
	img = Image.open( get_data('boat.png') )
	return np.asarray(img)


def get_images(exclude=None,colored=False):

	if colored:
		files = ['alley.png','building.png','man_color.png','flowers.png','girl.png']
	else:
		files = ['lena.png','barbara.png','man.png','boat.png','house.png','hill.png']
	if exclude is not None:
		files.remove(exclude)
	imgs = []
	for i in range(len(files)):

		img =  np.asarray(Image.open( get_data(files[i]) ) )
		imgs.append(img)
	return imgs


def monitor_progress(results=None,n_tasks=None,message=None):

	from time import time
	time_per_task = 0
	t_start = time()
	while True:
		complete_count = np.sum([1 for result in results if result.ready()])
		if complete_count == n_tasks:
			break
		duration = time() - t_start
		if complete_count != 0:
			time_per_task = duration / float(complete_count)
		incomplete_count = n_tasks - complete_count
		eta = incomplete_count * time_per_task
		mins, secs = divmod(eta, 60)
		hours, mins = divmod(mins, 60)
		#"%d:%02d:%02d" % (h, m, s)
		eta_string = " ,eta=%02d:%02d:%02d" % (hours,mins,secs)


		sys.stdout.write("\r"+message+":%3.2f%%" % (( complete_count / float(n_tasks))*100)+eta_string)
		sys.stdout.flush()
	print ""

def joblib_print(n_tasks,msg):
	def print_progress(self):
			sys.stdout.write("\r"+msg+":%3.2f%%" %
				(( self.n_completed_tasks / float(n_tasks))*100))
			sys.stdout.flush()
	return print_progress


def aggregate_proc(results,Z,i):
	Z[:,i] = results[i]

#it puts each element of results list into each
#column of Z
def aggregate(results,Z,n_jobs=2):

	from joblib import Parallel,delayed

	Parallel(n_jobs=n_jobs)(delayed(aggregate_proc)
						( results,Z,i) for i in range(len(results)))

	return Z
