
import os
import yaml
import numpy as np
import shutil
import lyssa
import warnings
from config import save_paths


max_folders = int(1e3)
#a dictionary mapping ids to folder_paths
workspace_map_file = os.path.join(save_paths[0],"workspace_map.yml")
if not os.path.isfile(workspace_map_file):
	open(workspace_map_file, 'wa').close()
workspace_map = None

#import pdb; pdb.set_trace()
with open(workspace_map_file, 'r') as handle:
	workspace_map = yaml.load(handle)
	if workspace_map is not None:
		#delete the entries with non-existed
		#folders
		delete_list = []
		for _id,folder in workspace_map.iteritems():

			if os.path.exists(folder):
				#if the folder exists but has no
				#data in it
				contents = os.listdir(folder)

				if len(contents) == 0:
					shutil.rmtree(folder)
					print "deleting",folder
					delete_list.append(_id)
			else:
				print "workspace",_id,"==>",folder
				print "not found"
				ans = raw_input('Do you want to delete the entry? (y|n):')
				if ans in ["y","yes","Y","Yes"]:
					print "deleting",folder
					delete_list.append(_id)


		for _id in delete_list:
			del workspace_map[_id]

if workspace_map is not None:
	with open(workspace_map_file, 'w') as handle:
		handle.write( yaml.dump(workspace_map, default_flow_style=False) )
else:
	workspace_map = {}
max_mmap_files = 100000
mmap_base = os.path.join(save_paths[0],'mmap_files/')
if not os.path.exists(mmap_base):
	os.makedirs(mmap_base)





def disk_usage(path):
	"""Return disk usage statistics about the given path.

	Returned valus is a named tuple with attributes 'total', 'used' and
	'free', which are the amount of total, used and free space, in bytes.
	"""
	st = os.statvfs(path)
	free = st.f_bavail * st.f_frsize
	#total = st.f_blocks * st.f_frsize
	used = (st.f_blocks - st.f_bfree) * st.f_frsize
	#measured in gigabytes
	free /= float(1024**3)
	used /= float(1024**3)

	return {"used":used, "free":free}


class workspace_manager():

	def __init__(self,base_path=None,metadata=None):
		if base_path is not None:
			#TODO:
			#the user may manually specify a location
			self.base_path = base_path
		else:
			_id,folder_location = request_workspace()
			print "-----------------------------------------"
			print "workspace created"
			print "id:",_id
			print "location:",folder_location
			print "-----------------------------------------"
			self.base_path = folder_location
		#self._id = _id
		self.metadata = metadata
		self.metadata_file = os.path.join(self.base_path,"meta.yml")
		if self.metadata is not None:
			self.set_metadata(self.metadata)
		else:
			self.get_metadata()

	def set_metadata(self,metadata):
		self.metadata = metadata
		with open(self.metadata_file, 'w') as handle:
			handle.write( yaml.dump(metadata, default_flow_style=False) )

	def get_metadata(self):
		if not os.path.exists(self.metadata_file):
			#print "no metadata file found on this workspace"
			return
		with open(self.metadata_file, 'r') as handle:
			#workspace_map = pickle.load(handle)
			self.metadata = yaml.load(handle)

	def show_metadata(self):
		#refresh
		self.get_metadata()
		print "--------------------------------------"
		for k,v in self.metadata.iteritems():
			print k,"==>",v
		print "--------------------------------------"


	def __del__(self):

		#print "cleaning up..."
		pass

	def add(self,path):
		#create an empty folder
		#under workspacepath
		if path[0] == "/":
			path = path[1:]
		fpath = os.path.join(self.base_path,path)
		if not os.path.exists(fpath):
			os.makedirs(fpath)

		return fpath

	def save(self,path,data):
		#if folder in the path do not exist,
		#they will be created
		#path can be like "/folder1/folder2/file.npy"
		#or "file.npy"
		#or "/folder1/folder2/" e.g /imgs which means that
		#the user wants to store the list data into the imgs folder
		#within the workspace folder
		from .dataset import save as dataset_save
		base,filename = os.path.split(path)
		fl = filename.split(".")
		prefix = ".".join(fl[:-1])
		suffix = fl[-1]
		#import pdb; pdb.set_trace()
		if base == "" or base == "/":
			#the path has only a filename
			#e.g dict.npy
			dataset_save(data,path=self.base_path,prefix=prefix,suffix="."+suffix)
		else:
			#the path contains directories
			if prefix != '':
				#case /folder/file.npy
				fbase = os.path.join(self.base_path,base)
				dataset_save(data,path=fbase,prefix=prefix,suffix="."+suffix)
			else:
				#e.g /folder1/imgs
				#the prefix is img
				#actually this should neves be called
				#and an online writer used instead
				prefix = suffix[:-1]
				fbase = os.path.join(self.base_path,path)
				dataset_save(data,path=fbase,prefix=prefix)

	def load(self,path,online=False,sparse=False):
		from .dataset import load as dataset_load
		from .dataset import get_mmap
		if path[0] == "/":
			path = path[1:]
		base,filename = os.path.split(path)
		full_path = os.path.join(self.base_path,path)
		return dataset_load(full_path,online=online,sparse=sparse)

	def get_writer(path,prefix="",n_files=10000,sparse=False):
		from .dataset import online_writer
		return online_writer(path=path,prefix=prefix,sparse=sparse,n_files=n_files)

	def contains(self,path):
		if path[0] == "/":
			path = path[1:]
		full_path = os.path.join(self.base_path,path)
		return os.path.isfile(full_path) or os.path.isdir(full_path)


def show_workspaces():

	print "--------------------------------------"

	for k,v in workspace_map.iteritems():
		print k,"==>",v
		workspace = get_workspace(id=k)
		workspace.show_metadata()
	print "--------------------------------------"


def clean_workspaces():

	delete_list = []
	for _id,folder in workspace_map.iteritems():
		if not os.path.exists(folder):
			delete_list.append(_id)

	for _id in delete_list:
		del workspace_map[_id]

	with open(workspace_map_file, 'w') as handle:
		handle.write( yaml.dump(workspace_map, default_flow_style=False) )

def copy_workspace(path=None,id=None):
	from distutils.dir_util import copy_tree
	new_id,new_folder_location = request_workspace()
	#find the folder location of the workspace you want
	#to copy from
	folder_location = None
	with open(workspace_map_file, 'r') as handle:
		workspace_map = yaml.load(handle)
		if id is not None:
			folder_location = workspace_map.get(id)
	if folder_location is None:
		warnings.warn("the provided id is not assigned to any location")

	copy_tree(folder_location,new_folder_location)
	return workspace_manager(base_path=new_folder_location)

def get_workspace(path=None,id=None):
	#return a workspace manager
	#given its the folder location or its id
	folder_location = None
	if id is not None:
		folder_location = workspace_map.get(id)
	if folder_location is None:
		warnings.warn("the provided id is not assigned to any location")

	return workspace_manager(base_path=folder_location)

def request_workspace(folder_name=None):

	#if folder_name is not None we name
	#it as specified

	fspace_list = []
	for save_path in save_paths:
		if not os.path.exists(save_path):
			fspace_list.append(0)
			continue
		du = disk_usage(save_path)
		fspace_list.append(du["free"])
	#return a folder location where
	#we have the most space
	base_save_path = save_paths[np.argmax(fspace_list)]

	curr_name_num = 0
	existing_folders = os.listdir(base_save_path)
	existing_folders.sort()
	#import pdb; pdb.set_trace()
	existing_folders = [folder for folder in existing_folders if folder.isdigit()]
	if len(existing_folders) == 0:
		name_num = 0
	for i in range(len(existing_folders)):
		#find appropriate location
		#i.e if the folders name are [0001,0003,0004]
		#you want to assign the new folder as 0002
		if i+1 == len(existing_folders):
			name_num = int(existing_folders[-1])+1
			break
		fnum = int(existing_folders[i])
		fnum_next = int(existing_folders[i+1])
		if curr_name_num < fnum or fnum < curr_name_num < fnum_next:
			name_num = curr_name_num
			break
		curr_name_num += 1

	_id = name_num
	base = (len(str(max_folders))-len(str(_id)))*'0'
	folder_name = str(base)+str(_id)
	folder_location = os.path.join(base_save_path,folder_name)
	if not os.path.exists(folder_location):
		os.makedirs(folder_location)
	wkeys = workspace_map.keys()
	if len(wkeys) > 0:
		workspace_id = int(np.sort(wkeys)[-1]+1)
	else:
		workspace_id = 0
	if workspace_map.get(workspace_id) is not None:
		warnings.warn("the allocated id refers to an existing folder")
		import pdb; pdb.set_trace()

	#print "adding {0}==>{1} in {2}".format(workspace_id,folder_location,workspace_map)
	workspace_map[workspace_id] = folder_location

	with open(workspace_map_file, 'w') as handle:
		handle.write( yaml.dump(workspace_map, default_flow_style=False) )

	return workspace_id,folder_location
