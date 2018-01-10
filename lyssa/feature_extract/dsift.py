from __future__ import division
import numpy as np
from scipy import signal
import math
from lyssa.utils import get_empty_mmap
from lyssa.utils.math import norm_cols
from lyssa.utils import joblib_print

'''
This module implements some basic functions that
does dense sift feature extraction. It follows the Matlab implementation
of Svetlana Lazebnik at http://www.cs.unc.edu/~lazebnik/
'''

# sift parameters
n_angles = 8
n_bins = 4
n_samples = n_bins ** 2
alpha = 9.0
angles = np.array(range(n_angles)) * 2.0 * np.pi / n_angles


def gen_dgauss(sigma):
    '''
    gradient of the gaussian on both directions.
    '''
    fwid = np.int(2 * np.ceil(sigma))
    G = np.array(range(-fwid, fwid + 1)) ** 2
    G = G.reshape((G.size, 1)) + G
    G = np.exp(- G / 2.0 / sigma / sigma)
    G /= np.sum(G)
    GH, GW = np.gradient(G)
    GH *= 2.0 / np.sum(np.abs(GH))
    GW *= 2.0 / np.sum(np.abs(GW))
    return GH, GW


class DsiftExtractor:
    '''
    The class extracts dense sift features.
    grid_spacing: the spacing for sampling dense descriptors
    patch_size: the size for each sift patch
    nrml_thres: low contrast normalization threshold
    sigma_edge: the standard deviation for the gaussian smoothing
        before computing the gradient
    sift_thres: sift thresholding
    '''

    def __init__(self, grid_spacing=None, patch_size=None, nrml_thres=1.0,
                 sigma_edge=0.8, sift_thres=0.2):

        self.gs = grid_spacing
        self.ps = patch_size
        self.nrml_thres = nrml_thres
        self.sigma = sigma_edge
        self.sift_thres = sift_thres
        # compute the weight contribution map
        sample_res = self.ps / np.double(n_bins)
        sample_p = np.array(range(self.ps))
        sample_ph, sample_pw = np.meshgrid(sample_p, sample_p)
        sample_ph.resize(sample_ph.size)
        sample_pw.resize(sample_pw.size)
        bincenter = np.array(range(1, n_bins * 2, 2)) / 2.0 / n_bins * self.ps - 0.5
        bincenter_h, bincenter_w = np.meshgrid(bincenter, bincenter)
        bincenter_h.resize((bincenter_h.size, 1))
        bincenter_w.resize((bincenter_w.size, 1))
        dist_ph = abs(sample_ph - bincenter_h)
        dist_pw = abs(sample_pw - bincenter_w)
        weights_h = dist_ph / sample_res
        weights_w = dist_pw / sample_res
        weights_h = (1 - weights_h) * (weights_h <= 1)
        weights_w = (1 - weights_w) * (weights_w <= 1)
        self.weights = weights_h * weights_w

    def process_image(self, image, positionNormalize=False):
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
        feat_arr: the feature array, each row is a feature
        positions: the positions of the features in the array
        '''

        image = image.astype(np.double)
        if image.ndim == 3:
            # convert to grayscale
            image = np.mean(image, axis=2)
        # compute the grids
        h, w = image.shape
        gs = self.gs
        ps = self.ps
        rem_h = np.mod(h - ps, gs)
        rem_w = np.mod(w - ps, gs)
        from math import floor
        # the locations of the top left pixel
        # of each patch
        offset_h = int(floor(rem_h / 2.))
        offset_w = int(floor(rem_w / 2.))
        grid_h, grid_w = np.meshgrid(range(offset_h, h - ps + 1, gs), range(offset_w, w - ps + 1, gs))
        grid_h = grid_h.flatten()
        grid_w = grid_w.flatten()
        feat_arr = self.extract_sift_patches(image, grid_h, grid_w)
        if positionNormalize:
            positions = np.vstack((grid_h / np.double(h), grid_w / np.double(w)))
        else:
            positions = np.vstack((grid_h, grid_w))
        # the positions of the pixels in the top-left corner of the patch
        # positions[0,:] contains the heights
        # positions[1,:] contains the widths
        # feaArr contains each descriptor in a row
        return feat_arr, positions

    def extract_sift_patches(self, image, grid_h, grid_w):
        """extracts the sift descriptor of patches
           in positions (grid_h, grid_w) in the image"""
        h, w = image.shape
        n_patches = grid_h.size
        feat_arr = np.zeros((n_patches, n_samples * n_angles))

        # calculate gradient
        gh, gw = gen_dgauss(self.sigma)
        ih = signal.convolve2d(image, gh, mode='same')
        iw = signal.convolve2d(image, gw, mode='same')
        i_mag = np.sqrt(ih ** 2 + iw ** 2)
        i_theta = np.arctan2(ih, iw)
        i_orient = np.zeros((n_angles, h, w))
        for i in range(n_angles):
            i_orient[i] = i_mag * np.maximum(np.cos(i_theta - angles[i]) ** alpha, 0)
        for i in range(n_patches):
            curr_feature = np.zeros((n_angles, n_samples))
            for j in range(n_angles):
                curr_feature[j] = np.dot(self.weights, i_orient[j, grid_h[i]:grid_h[i] + self.ps,
                                                      grid_w[i]:grid_w[i] + self.ps].flatten())
            feat_arr[i] = curr_feature.flatten()
        # feaArr contains each descriptor in a row
        feat_arr = self.normalize_sift(feat_arr)
        return feat_arr

    def normalize_sift(self, feat_arr):
        '''
        This function does sift feature normalization
        following David Lowe's definition (normalize length ->
        thresholding at 0.2 -> renormalize length)
        '''
        siftlen = np.sqrt(np.sum(feat_arr ** 2, axis=1))
        hcontrast = (siftlen >= self.nrml_thres)
        siftlen[siftlen < self.nrml_thres] = self.nrml_thres
        # normalize with contrast thresholding
        feat_arr /= siftlen.reshape((siftlen.size, 1))
        # suppress large gradients
        feat_arr[feat_arr > self.sift_thres] = self.sift_thres
        # renormalize high-contrast ones
        feat_arr[hcontrast] /= np.sqrt(np.sum(feat_arr[hcontrast] ** 2, axis=1)). \
            reshape((feat_arr[hcontrast].shape[0], 1))
        return feat_arr


def _sift_extract_proc(img, n_desc_per_image, patch_shape):
    # runs in a separate process that extracts
    # n_desc_per_image sift descriptors from an image
    psize = patch_shape[0]
    # find the spacing when we use n_desc_per_image number
    # of patches per image
    spaceing = max(1, int(math.floor(math.sqrt(
        float(np.prod(img.shape)) / n_desc_per_image))))

    extractor = DsiftExtractor(grid_spacing=spaceing, patch_size=psize)
    features, _ = extractor.process_image(img, positionNormalize=False)
    return features.T


class sift_extractor:
    """extract n_descriptors SIFT descriptors from a database of images by extracting
       (n_descriptors / n_images) number of SIFT descriptors from each image
       the extracted SIFT come only from the full size image. We don't reduce the scale"""
    def __init__(self, n_descriptors=None, patch_shape=None, n_jobs=1, mmap=False):
        self.n_descriptors = n_descriptors
        self.patch_shape = patch_shape
        self.n_jobs = n_jobs
        self.mmap = mmap

    def __call__(self, imgs):
        """imgs: a list of 2D image arrays"""

        from joblib import Parallel, delayed
        n_imgs = len(imgs)
        n_desc_per_image = int(self.n_descriptors / float(n_imgs))
        # Z = run_parallel(func=_sift_extract_proc,data=imgs,args=(n_desc_per_image,self.patch_shape),
        #			result_shape=(n_features,n_imgs),n_batches=100,mmap=self.mmap,msg="building ScSPM features",n_jobs=n_jobs)
        if self.n_jobs > 1:
            msg = "extracting dsift"
            Parallel.print_progress = joblib_print(n_imgs, msg)
            results = Parallel(n_jobs=self.n_jobs)(delayed(_sift_extract_proc)
                                                   (imgs[i], n_desc_per_image, self.patch_shape) for i in range(n_imgs))
            n_descs = [results[i].shape[1] for i in range(len(results))]
            if self.mmap:
                Z = get_empty_mmap((results[0].shape[0], np.sum(n_descs)))
            else:
                Z = np.zeros((results[0].shape[0], np.sum(n_descs)))
            base = 0
            for j in range(n_imgs):
                offset = n_descs[j]
                Z[:, base:base + offset] = results[j]
                base += offset
        # normalize each SIFT descriptor
        Z = norm_cols(Z)

        return Z
