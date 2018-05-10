import numpy as np
from itertools import product
import sys
from lyssa.utils import run_parallel,get_empty_mmap
from PIL import Image
from functools import partial


def resize_img(img_arr, scale=None, maxdim=None):
    """
    if maxdim is not none
    resizes the image such that width and height are less than
    maxdim(preserving aspect ratio)
    """
    if maxdim is not None:
        imdim = max(img_arr.shape[0], img_arr.shape[1])
        if imdim > maxdim:
            scaler = float(maxdim) / float(imdim)
            new_h = int(round(scaler*img_arr.shape[0]))
            new_w = int(round(scaler*img_arr.shape[1]))
    elif scale is not None:
        new_h = int(round(scale*img_arr.shape[0]))
        new_w = int(round(scale*img_arr.shape[1]))
    import PIL
    img = PIL.Image.fromarray(np.uint8(img_arr))
    img = img.resize((new_w, new_h))
    img_arr  = np.array(img)
    return img_arr


def horizontal_reflection(img):
    #flips an image horizontally
    if isinstance(img,(np.ndarray, np.core.memmap)):
        img = np.array(Image.fromarray(img).transpose(Image.FLIP_LEFT_RIGHT))
    else:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    return img



"""
def rgb_vec_to_img(img_vec,patch_shape,normalized=False):

    n_channels = 3
    n_pixels = patch_shape[0]*patch_shape[1]
    red_img = img_vec[:n_pixels]
    green_img = img_vec[n_pixels:2*n_pixels]
    blue_img = img_vec[2*n_pixels:]

    if normalized:
        #scale to the range [0,255]
        from sklearn.preprocessing import MinMaxScaler
        mms = MinMaxScaler(feature_range=(0,255))
        red_img = mms.fit_transform(red_img.reshape(-1, 1))
        green_img = mms.fit_transform(green_img.reshape(-1, 1))
        blue_img = mms.fit_transform(blue_img.reshape(-1, 1))

    img = np.array(np.zeros((n_pixels,n_channels)))
    for i in range(n_pixels):
        img[i,0] = red_img[i]
        img[i,1] = green_img[i]
        img[i,2] = blue_img[i]

    img = img.reshape((patch_shape[0],patch_shape[1],n_channels))
    return img
"""


def scale_img(img):

    if np.max(img) > 1:
        return img / 255.
    else:
        return img


def MSE(reconstructed, image):
    reconstructed = scale_img(reconstructed)
    image = scale_img(image)
    return (1 / float(image.size)) * np.sum( np.power(reconstructed - image,2) )

def PSNR(reconstructed,image,MAX_I=1):
    #MAX_I is the maximum pixel value in the image
    #so we it is 1.0 for scaled images
    reconstructed = scale_img(reconstructed)
    image = scale_img(image)
    return 20 * np.log10(MAX_I) - 10 * np.log10(MSE(reconstructed,image))


def to_numpy_img(img):
    #correct the numpy RGB img
    #i.e 255+1=0 but we want 255+1=255
    zero_idx = np.where(img < 0)
    max_idx = np.where(img > 255)
    for i in range(len(zero_idx[0])):
        img[zero_idx[0][i],zero_idx[1][i],zero_idx[2][i]] = 0
    for i in range(len(max_idx[0])):
        img[max_idx[0][i],max_idx[1][i],max_idx[2][i]] = 255
    return np.uint8(img)


def add_noise(img,std,normalized=False):
    #adds noise to each channel of an image
    #for RGB images
    if img.ndim == 3:
        h,w,c = img.shape
        noise = np.random.normal(0.,std,(h,w,c))
        img = img + noise
        zero_idx = np.where(img < 0)
        max_idx = np.where(img > 255)

        #correct the numpy arithmetic
        #i.e 255+1=0 but we want 255+1=255
        for i in range(len(zero_idx[0])):
            img[zero_idx[0][i],zero_idx[1][i],zero_idx[2][i]] = 0
        for i in range(len(max_idx[0])):
            img[max_idx[0][i],max_idx[1][i],max_idx[2][i]] = 255

        return img
    elif img.ndim == 2:
        #assumes that each pixel is in [0,1]. If it is [0,255] use the correction as above.
        h,w = img.shape
        noise = np.random.normal(0.,std,(h,w))
        img = img + noise

        if normalized:
            zero_idx = np.where(img < 0)
            max_idx = np.where(img > 1)
            #clip so that we are in the scale [0,1]
            for i in range(len(zero_idx[0])):
                img[zero_idx[0][i],zero_idx[1][i]] = 0
            for i in range(len(max_idx[0])):
                img[max_idx[0][i],max_idx[1][i]] = 1
            return img
        else:
            zero_idx = np.where(img < 0)
            max_idx = np.where(img > 255)
            #clip so that we are in the scale [0,1]
            for i in range(len(zero_idx[0])):
                img[zero_idx[0][i],zero_idx[1][i]] = 0
            for i in range(len(max_idx[0])):
                img[max_idx[0][i],max_idx[1][i]] = 255
            return img


def generate_pepper(img,prob,return_mask=False):
    #prob is the proportion of missing pixels in the image
    #it is not the probability of setting a pixel to black
    from itertools import product
    h,w = img.shape[:2]
    corrupted_img = np.copy(img)
    pidx = list(product(range(h),range(w)))
    n_missing = int(prob * h* w)
    miss_idx = np.random.choice(range(len(pidx)),size=n_missing,replace=False)
    for i in range(n_missing):
        m_h,m_w = pidx[miss_idx[i]]
        if img.ndim == 2:
            corrupted_img[m_h,m_w] = 0
        elif img.ndim == 3:
            corrupted_img[m_h,m_w,:] = 0
    if return_mask:
        if img.ndim == 2:
            mask = np.zeros((img.shape[:2]))
        elif img.ndim == 3:
            mask = np.zeros((img.shape))
        for i in range(h):
            for j in range(w):
                if img.ndim == 2:
                    if corrupted_img[i,j] > 0.:
                        mask[i,j] = 1
                elif img.ndim == 3:
                    pass
        return corrupted_img,mask
    else:
        return corrupted_img

"""
def reconstruct_from_patches(X,patch_shape,image_shape,noisy_img=None,weight=None):

    #X : contains the patches in the columns.
    #Each patch is a vector of length (patch_shape[0] * patch_shape[1]) in the grayscale case
    #while in the rgb case it is a concatenated RGB vector of shape (patch_shape[0] * patch_shape[1] * 3)

    #if noisy_img is not None then we want:
    #the value of a pixel in the denoised image to be computed by averaging the value of this pixel in
    #the noisy image (weighted by "weight") and the values of this pixel on the patches to which it belongs
    #(weighted by 1)


    #the grayscale case
    if len(image_shape) == 2:
        i_h, i_w = image_shape
        p_h, p_w = patch_shape
        img = np.zeros(image_shape)
        # compute the dimensions of the patches array
        n_h = i_h - p_h + 1
        n_w = i_w - p_w + 1
        n_patches = X.shape[1]
        patch_idx = range(n_patches)
        for n, (i, j) in zip(patch_idx, product(range(n_h), range(n_w))):
            patch = X[:,n].reshape((patch_shape))
            img[i:i + p_h, j:j + p_w] += patch

        #add a weighted constribution of the noisy image
        if noisy_img is not None:
            img = img + weight*noisy_img

        #each pixel is the averages of all the patches it
        #is in
        #TODO: replace by a weighted average
        for i in range(i_h):
            for j in range(i_w):
                # divide by the amount of overlap
                norm_const = min(i + 1, p_h, i_h - i) * min(j + 1, p_w, i_w - j)
                if weight is not None:
                    norm_const += weight
                #import pdb
                #pdb.set_trace()
                img[i,j] = img[i,j] / float(norm_const)

        return img

    elif len(image_shape) == 3:
        i_h, i_w = image_shape[:2]
        p_h, p_w = patch_shape
        img = np.zeros(image_shape)
        # compute the dimensions of the patches array
        n_h = i_h - p_h + 1
        n_w = i_w - p_w + 1
        n_patches = X.shape[1]
        patch_idx = range(n_patches)
        for n, (i, j) in zip(patch_idx, product(range(n_h), range(n_w))):
            #patch = rgb_vec_to_img(X[:,n],patch_shape,normalized=True).astype(float)
            patch = rgb_vec_to_img(X[:,n],patch_shape).astype(float)
            img[i:i + p_h, j:j + p_w,:] += patch
            #import pdb
            #pdb.set_trace()
        #add a weighted constribution of the noisy image
        if noisy_img is not None:
            img = img + weight*noisy_img
        #each pixel is the averages of all the patches it
        #is in
        #TODO: replace by a weighted average
        for i in range(i_h):
            for j in range(i_w):
                # divide by the amount of overlap
                norm_const = min(i + 1, p_h, i_h - i) * min(j + 1, p_w, i_w - j)
                if weight is not None:
                    norm_const += weight
                img[i,j,:] = img[i,j,:] / float(norm_const)

        return img


"""


def compute_n_patches(h,w,patch_size,step_size,padding=False):
    if padding:
        padding_w = int(np.floor(patch_size/2))
        padding_h = int(np.floor(patch_size/2))
    else:
        padding_w = 0
        padding_h = 0

    n_patches_h = ( (h-patch_size+padding_h) // step_size ) + 1
    n_patches_w = ( (w-patch_size+padding_w) // step_size ) + 1

    #for pooling:
    #patch_size is the pooling size and pooling_step
    #is step_size and the above formula holds without
    #the padding
    return (n_patches_h,n_patches_w)



def n_dataset_patches(imgs,patch_size=None,patches_per_image=None,step_size=None):

    if patches_per_image is not None:
        step_size = 1

    n_imgs = len(imgs)
    n_data_patches = np.zeros(n_imgs).astype(int)

    for i in range(n_imgs):
        h,w = imgs[i].shape[:2]
        #loc_h,loc_w = grid_patches_loc(h,w, patch_size, step_size)
        n_patches_h,n_patches_w = compute_n_patches(h,w,patch_size,step_size,padding=False)
        npatches = n_patches_h*n_patches_w
        if patches_per_image is not None:
            #sample patches randomly
            if patches_per_image < npatches:
                npatches = patches_per_image

        n_data_patches[i] = npatches

    return n_data_patches


def extract_patches(imgs,step_size=None,n_patches=None,patch_size=None,mmap=False,scale=False,verbose=False,mem="high",n_jobs=1):

    #extracts n_patches from a set of images.
    #It calls grid_patches with a specific spacing.
    #if patch_shape = (a,a) then patch_size = a.
    #imgs is a list of 2D images

    n_imgs = len(imgs)
    if n_patches is not None:
        patches_per_image = int(np.floor(float(n_patches)/float(n_imgs)))
        print "using {0} patches per image".format(patches_per_image)
    #find the number of actual patches
    #each image has
    if n_patches is not None:
        patch_numbers = n_dataset_patches(imgs,patch_size=patch_size,patches_per_image=patches_per_image)
    else:
        patch_numbers = n_dataset_patches(imgs,patch_size=patch_size,step_size=step_size)

    print "total number of patches {0}".format(np.sum(patch_numbers))
    if mem == "high":

        import multiprocessing
        pool = multiprocessing.Pool(processes=n_jobs,initializer=None)
        results = []
        for i in range(n_imgs):
            if verbose:
                sys.stdout.write("\rextracting patches:%3.2f%%" % (( i / float(n_imgs))*100))
                sys.stdout.flush()
            if n_patches is not None:
                func = partial(grid_patches,patch_size=patch_size,n_patches=patches_per_image,scale=scale)
            else:
                func = partial(grid_patches,patch_size=patch_size,step_size=step_size,scale=scale)
            results.append(pool.apply_async(func, (imgs[i],)))

        pool.close() # no more tasks
        if verbose:
            sys.stdout.write("\rextracting patches:%3.2f%%" % (100))
            sys.stdout.flush()
            print ""
        n_patches = np.sum(patch_numbers)
        if mmap:
            patches = get_empty_mmap((results[0].get().shape[0],n_patches))
            print "things are mmaped!"
        else:
            patches = np.zeros((results[0].get().shape[0],n_patches))

        base = 0
        for i in range(n_imgs):
            result = results[i].get()
            results[i] = None
            offset = patch_numbers[i]
            patches[:,base:base+offset] = result
            base += offset

    else:
        n_patches = np.sum(patch_numbers)
        if len(imgs[0].shape) == 2:
            patches = np.zeros((patch_size**2,n_patches))
        elif len(imgs[0].shape) == 3:
            patches = np.zeros(( imgs[0].shape[2]*(patch_size**2),n_patches))

        base = 0
        for i in range(n_imgs):
            if verbose:
                sys.stdout.write("\rextracting patches:%3.2f%%" % (( i / float(n_imgs))*100))
                sys.stdout.flush()
            if n_patches is not None:
                _patches = grid_patches(imgs[i],patch_size=patch_size,n_patches=patches_per_image,scale=scale)
            else:
                _patches = grid_patches(imgs[i],patch_size=patch_size,step_size=step_size,scale=scale)
            offset = patch_numbers[i]
            patches[:,base:base+offset] = _patches
            base += offset

    if step_size is not None:
        return patches,patch_numbers
    return patches


def extract_patches_loc(imgs,n_patches,patch_size):

    n_imgs = len(imgs)
    patches_per_image = int(round(float(n_patches)/float(n_imgs)))
    if patches_per_image == 0:
        print "insufficient amount of patches"
    locs = []
    for i in range(n_imgs):
        img = imgs[i]
        h,w = img.shape[:2]
        step = int(round(w *  patches_per_image**(-0.5)))
        loc_h,loc_w = grid_patches_loc(img, patch_size, step)
        locs.append([loc_h,loc_w])

    return locs



def grid_patches_loc(h,w,patch_size,step_size):
    """ This is exactly the same as grid_patches but it returns only the locations of the patches
        and not the patches themselves
    """

    spaceX = range(0, w-patch_size+1, step_size)
    spaceY = range(0, h-patch_size+1, step_size)
    npatches = len(spaceX)*len(spaceY)
    #the position of the top-left pixel of each patch
    loc_h = np.zeros(npatches).astype(int)
    loc_w = np.zeros(npatches).astype(int)
    # Extract the patches and get the patch centres
    cnt = 0
    #height
    for sy in spaceY:
        #width
        for sx in spaceX:
            loc_h[cnt] = sy
            loc_w[cnt] = sx
            cnt +=1

    return loc_h, loc_w








def grid_patches(img, patch_size=None, step_size=None,n_patches=None,return_loc=False,scale=False):


    """
    extract a grid of (overlapping) patches from an image
    as a 2D matrix of shape (n_rows*n_cols,n_patches)
    """


    from numpy.lib.stride_tricks import as_strided
    if not isinstance(img,(np.ndarray,np.core.memmap)):
        img = np.array(img)

    n_req_patches = n_patches
    # Check and get image dimensions
    if img.ndim == 3:
        (Ih, Iw, Ic) = img.shape
        patch_shape = (patch_size,patch_size,Ic)
    elif img.ndim == 2:
        (Ih, Iw) = img.shape
        img = img.reshape((Ih,Iw,-1))
        Ic = 1
        patch_shape = (patch_size,patch_size,Ic)
    else:
        raise ValueError('image must be a 2D or 3D np.array')

    if n_req_patches is not None:
        step_size = 1


    slices = [slice(None, None, step_size),slice(None, None, step_size),slice(None, None, step_size)]
    n_patches_h,n_patches_w = compute_n_patches(Ih,Iw,patch_size,step_size,padding=False)
    n_patches = n_patches_h*n_patches_w

    patch_strides = img.strides
    indexing_strides = img[slices].strides
    #patch_indices_shape = np.array([n_patches_h,n_patches_w])
    patch_indices_shape = ((np.array(img.shape) - np.array(patch_shape)) //
                       np.array(step_size)) + 1

    shape = tuple(list(patch_indices_shape) + list(patch_shape))
    strides = tuple(list(indexing_strides) + list(patch_strides))

    patches = as_strided(img, shape=shape, strides=strides)

    if Ic == 1:
        patches = patches.reshape((n_patches_h,n_patches_w, patch_size*patch_size))
        patches = patches.reshape((n_patches_h*n_patches_w, -1)).T
    else:
        patches = patches.reshape((n_patches_h,n_patches_w, patch_size*patch_size*Ic))
        patches = patches.reshape((n_patches_h*n_patches_w, -1)).T

    if n_req_patches is not None:
        if n_req_patches < n_patches:
            if Ic <= 3:
                mean_intensity = np.mean(patches,axis=0)
                good_patches = np.arange(n_patches)
                patch_idxs = np.random.choice(good_patches,n_req_patches,replace=False)
            else:
                patch_idxs = np.random.choice(np.arange(n_patches),n_req_patches,replace=False)
            patches = patches[:,patch_idxs]

    return patches
