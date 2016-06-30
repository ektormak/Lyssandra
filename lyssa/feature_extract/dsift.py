
from __future__ import division
import numpy as np
from scipy import signal
import math
from lyssa.utils import run_parallel,get_empty_mmap

'''
This module implements some basic functions that
does dense sift feature extraction. It follows the Matlab implementation
of Svetlana Lazebnik at http://www.cs.unc.edu/~lazebnik/
'''

#sift parameters
Nangles = 8
Nbins = 4
Nsamples = Nbins**2
alpha = 9.0
angles = np.array(range(Nangles))*2.0*np.pi/Nangles

def gen_dgauss(sigma):
	'''
	gradient of the gaussian on both directions.
	'''
	fwid = np.int(2*np.ceil(sigma))
	G = np.array(range(-fwid,fwid+1))**2
	G = G.reshape((G.size,1)) + G
	G = np.exp(- G / 2.0 / sigma / sigma)
	G /= np.sum(G)
	GH,GW = np.gradient(G)
	GH *= 2.0/np.sum(np.abs(GH))
	GW *= 2.0/np.sum(np.abs(GW))
	return GH,GW

class DsiftExtractor:
	'''
	The class extracts dense sift features.
	gridSpacing: the spacing for sampling dense descriptors
	patchSize: the size for each sift patch
	nrml_thres: low contrast normalization threshold
	sigma_edge: the standard deviation for the gaussian smoothing
		before computing the gradient
	sift_thres: sift thresholding
	'''
	def __init__(self, gridSpacing=None, patchSize=None,
				 nrml_thres = 1.0,\
				 sigma_edge = 0.8,\
				 sift_thres = 0.2):

		self.gS = gridSpacing
		self.pS = patchSize
		self.nrml_thres = nrml_thres
		self.sigma = sigma_edge
		self.sift_thres = sift_thres
		# compute the weight contribution map
		sample_res = self.pS / np.double(Nbins)
		sample_p = np.array(range(self.pS))
		sample_ph, sample_pw = np.meshgrid(sample_p,sample_p)
		sample_ph.resize(sample_ph.size)
		sample_pw.resize(sample_pw.size)
		bincenter = np.array(range(1,Nbins*2,2)) / 2.0 / Nbins * self.pS - 0.5
		bincenter_h, bincenter_w = np.meshgrid(bincenter,bincenter)
		bincenter_h.resize((bincenter_h.size,1))
		bincenter_w.resize((bincenter_w.size,1))
		dist_ph = abs(sample_ph - bincenter_h)
		dist_pw = abs(sample_pw - bincenter_w)
		weights_h = dist_ph / sample_res
		weights_w = dist_pw / sample_res
		weights_h = (1-weights_h) * (weights_h <= 1)
		weights_w = (1-weights_w) * (weights_w <= 1)
		self.weights = weights_h * weights_w

	def process_image(self, image, positionNormalize = False):
		'''
		processes a single image, return the locations
		and the values of detected SIFT features.
		image: a M*N image which is a numpy 2D array. If you
			pass a color image, it will automatically be converted
			to a grayscale image.
		positionNormalize: whether to normalize the positions
			to [0,1]. If False, the pixel-based positions of the
			top-right position of the patches is returned.

		Return values:
		feaArr: the feature array, each row is a feature
		positions: the positions of the features
		'''

		image = image.astype(np.double)
		if image.ndim == 3:
			#convert to grayscale
			image = np.mean(image,axis=2)
		# compute the grids
		H,W = image.shape
		gS = self.gS
		pS = self.pS
		remH = np.mod(H-pS, gS)
		remW = np.mod(W-pS, gS)
		from math import floor
		#the locations of the top left pixel
		#of each patch
		offsetH = int(floor(remH/2.))
		offsetW = int(floor(remW/2.))
		gridH,gridW = np.meshgrid(range(offsetH,H-pS+1,gS), range(offsetW,W-pS+1,gS))
		gridH = gridH.flatten()
		gridW = gridW.flatten()
		feaArr = self.calculate_sift_grid(image,gridH,gridW)
		feaArr = self.normalize_sift(feaArr)
		if positionNormalize:
			positions = np.vstack((gridH / np.double(H), gridW / np.double(W)))
		else:
			positions = np.vstack((gridH, gridW))
		#the positions of the pixels in the top-left corner of the patch
		#positions[0,:] contains the heights
		#positions[1,:] contains the widths
		#feaArr contains each descriptor in a row
		return feaArr, positions

	def calculate_sift_grid(self,image,gridH,gridW):

		H,W = image.shape
		Npatches = gridH.size
		feaArr = np.zeros((Npatches,Nsamples*Nangles))

		# calculate gradient
		GH,GW = gen_dgauss(self.sigma)
		IH = signal.convolve2d(image,GH,mode='same')
		IW = signal.convolve2d(image,GW,mode='same')
		Imag = np.sqrt(IH**2+IW**2)
		Itheta = np.arctan2(IH,IW)
		Iorient = np.zeros((Nangles,H,W))
		for i in range(Nangles):
			Iorient[i] = Imag * np.maximum(np.cos(Itheta - angles[i])**alpha,0)
		for i in range(Npatches):
			currFeature = np.zeros((Nangles,Nsamples))
			for j in range(Nangles):
				currFeature[j] = np.dot(self.weights,\
						Iorient[j,gridH[i]:gridH[i]+self.pS, gridW[i]:gridW[i]+self.pS].flatten())
			feaArr[i] = currFeature.flatten()
		return feaArr


	def extract_sift_patches(self,image,gridH,gridW):
		#extracts the sift descriptor of patches
		#in positions (gridH,gridW) in the image
		H,W = image.shape
		Npatches = gridH.size
		feaArr = np.zeros((Npatches,Nsamples*Nangles))

		# calculate gradient
		GH,GW = gen_dgauss(self.sigma)
		IH = signal.convolve2d(image,GH,mode='same')
		IW = signal.convolve2d(image,GW,mode='same')
		Imag = np.sqrt(IH**2+IW**2)
		Itheta = np.arctan2(IH,IW)
		Iorient = np.zeros((Nangles,H,W))
		for i in range(Nangles):
			Iorient[i] = Imag * np.maximum(np.cos(Itheta - angles[i])**alpha,0)
		for i in range(Npatches):
			currFeature = np.zeros((Nangles,Nsamples))
			for j in range(Nangles):
				currFeature[j] = np.dot(self.weights,\
						Iorient[j,gridH[i]:gridH[i]+self.pS, gridW[i]:gridW[i]+self.pS].flatten())
			feaArr[i] = currFeature.flatten()
		#feaArr contains each descriptor in a row
		feaArr = self.normalize_sift(feaArr)
		return feaArr

	def normalize_sift(self,feaArr):
		'''
		This function does sift feature normalization
		following David Lowe's definition (normalize length ->
		thresholding at 0.2 -> renormalize length)
		'''
		siftlen = np.sqrt(np.sum(feaArr**2,axis=1))
		hcontrast = (siftlen >= self.nrml_thres)
		siftlen[siftlen < self.nrml_thres] = self.nrml_thres
		# normalize with contrast thresholding
		feaArr /= siftlen.reshape((siftlen.size,1))
		# suppress large gradients
		feaArr[feaArr>self.sift_thres] = self.sift_thres
		# renormalize high-contrast ones
		feaArr[hcontrast] /= np.sqrt(np.sum(feaArr[hcontrast]**2,axis=1)).\
				reshape((feaArr[hcontrast].shape[0],1))
		return feaArr

"""
class SingleSiftExtractor(DsiftExtractor):
	'''
	The simple wrapper class that does feature extraction, treating
	the whole image as a local image patch.
	'''
	def __init__(self, patchSize,
				 nrml_thres = 1.0,\
				 sigma_edge = 0.8,\
				 sift_thres = 0.2):
		# simply call the super class __init__ with a large gridSpace
		DsiftExtractor.__init__(self, patchSize, patchSize, nrml_thres, sigma_edge, sift_thres)

	def process_image(self, image):
		return DsiftExtractor.process_image(self, image, False, False)[0]

"""





#used as a process that extracts n_desc_per_image sift descriptors
#from an image
def _sift_extract_proc(img,n_desc_per_image,patch_shape):



	#sift descriptor dimensions
	n_features = 128
	Z = np.empty((n_features , n_desc_per_image))

	h,w = img.shape[:2]
	psize = patch_shape[0]

	#find the spaceing when we use n_desc_per_image number
	#of patches per image
	spaceing = max(1, int(math.floor(math.sqrt( \
		float(np.prod(img.shape))/n_desc_per_image))))

	extractor = DsiftExtractor(gridSpacing=spaceing,patchSize=psize)
	features,_ = extractor.process_image(img,positionNormalize=False)
	return features.T

#extract n_descriptors SIFT descriptors from a database of images
#extract (n_descriptors / n_images) number of SIFT descriptors from each image
#NOTE: we extract SIFT only from the full size image and we dont reduce the scale
class sift_extractor:
	def __init__(self,n_descriptors=None,patch_shape=None,n_jobs=1,mmap=False):
		self.n_descriptors = n_descriptors
		self.patch_shape = patch_shape
		self.n_jobs = n_jobs
		self.mmap = mmap
	def __call__(self,imgs):
		#imgs is a list of 2D image arrays

		from joblib import Parallel, delayed
		from lyssa.utils import gen_even_batches

		n_imgs = len(imgs)
		n_desc_per_image = int(self.n_descriptors / float(n_imgs) )
		#Z = run_parallel(func=_sift_extract_proc,data=imgs,args=(n_desc_per_image,self.patch_shape),
		#			result_shape=(n_features,n_imgs),n_batches=100,mmap=self.mmap,msg="building ScSPM features",n_jobs=n_jobs)

		if self.n_jobs > 1:
			from lyssa.utils import joblib_print
			msg = "extracting dsift"
			Parallel.print_progress = joblib_print(n_imgs,msg)
			results = Parallel(n_jobs=self.n_jobs)(delayed(_sift_extract_proc)
						( imgs[i] ,n_desc_per_image ,self.patch_shape) for i in range(n_imgs))
			n_descs = [results[i].shape[1] for i in range(len(results))]
			if self.mmap:
				Z = get_empty_mmap((results[0].shape[0],np.sum(n_descs)))
			else:
				Z = np.zeros((results[0].shape[0],np.sum(n_descs)))
			base = 0
			for j in range(n_imgs):
				offset = n_descs[j]
				Z[:,base:base+offset] = results[j]
				base += offset


		#normalize each SIFT descriptor
		from lyssa.utils.math import norm_cols
		Z = norm_cols(Z)

		return Z
