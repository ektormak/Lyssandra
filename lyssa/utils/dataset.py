import numpy as np
import os
import sys
import gc
import PIL
import os.path
import shutil
from .workspace import mmap_base
from .workspace import max_mmap_files
import pickle
from scipy import sparse


def save_sparse_matrix(filename, X):
    """save the sparse matrix X as filename + ".npz"
    """
    X_coo = X.tocoo()
    row = X_coo.row
    col = X_coo.col
    data = X_coo.data
    shape = X_coo.shape
    np.savez(filename, row=row, col=col, data=data, shape=shape)


def load_sparse_matrix(filename, shape=None):
    # filename should end in .npz
    X = np.load(filename)
    X = sparse.coo_matrix((X['data'], (X['row'], X['col'])), shape=X['shape'])
    # return as a dense matrix
    if shape is not None:
        return np.array(X.todense()).reshape(shape)
    else:
        return np.array(X.todense())


def save(data, path=None, prefix=None, suffix=".npy"):
    """
    saves the dataset to the specified path
    the dataset can either be a numpy array
    or a list of numpy arrays(each might have a different size)
    """
    if not os.path.exists(path):
        os.makedirs(path)

    if type(data) is np.ndarray or type(data) is np.core.memmap:

        if suffix == ".npy":
            np.save(os.path.join(path, prefix+suffix), data)
        elif suffix == ".npz":
            # compress and save
            pass
    elif type(data) is list or hasattr(data, '__len__'):
        # case it is a proxy like the online reader class
        n_files = len(data)
        for i in range(n_files):
            # to preserve ordering when reading the files
            base = (len(str(n_files))-len(str(i)))*'0'
            formated_index = str(base)+str(i)
            np.save(os.path.join(path, prefix+formated_index+suffix), data[i])

    if suffix == ".pickle":
        # assume it is a custom object so pickle it
        with open(os.path.join(path, prefix+suffix), 'wb') as handle:
            pickle.dump(data, handle)


def load(path, online=False, sparse=False):
    """
    load the data in <path>,
    assumes that the data files in that path are of the same type
    """
    if os.path.isfile(path):
        _, suffix = os.path.splitext(path)
        if suffix == ".npy":
            data = np.load(path)
            if online:
                data = get_mmap(data)
            return data
        elif suffix == ".pickle":
            with open(path, 'rb') as handle:
                data = pickle.load(handle)
            return data
    else:
        files = [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        if online:
            return online_reader(data_files=files, sparse=sparse)
        else:
            data = []
            files.sort()
            for i in range(len(files)):
                data.append(np.load(files[i]))
            return data


def to_data_matrix(path, mmap=False):
    """
    converts a list of numpy array vectors into a data matrix
    assumes that the vectors have the same length
    """
    files = [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    n_samples = len(files)
    n_features = np.load(files[0]).shape[0]
    X = np.zeros((n_samples,n_features))
    for i in range(n_samples):
        X[i,:] = np.load(files[i])
    if mmap:
        return get_mmap(X)
    else:
        return X


def get_mmap(X):
    """
    converts a numpy array to
    a numpy memmory mapped array
    """
    # TODO: use tempfile.NamedTemporaryFile
    if type(X) is np.core.memmap:
        return X
    fid = 0
    filename = mmap_base+"data"+str(fid)+".dat"
    for i in range(max_mmap_files):
        if os.path.isfile(filename):
            fid += 1
            filename = mmap_base+"data"+str(fid)+".dat"
        else:
            break

    _X = np.memmap(filename, dtype='float64', mode='w+', shape=X.shape)
    _X[:] = X[:]
    del X
    gc.collect()
    return _X


def get_empty_mmap(shape):
    """
    creates an empty memmory mapped array
    """
    fid = 0
    filename = mmap_base+"data"+str(fid)+".dat"
    for i in range(max_mmap_files):
        if os.path.isfile(filename):
            fid += 1
            filename = mmap_base+"data"+str(fid)+".dat"
        else:
            break

    return np.memmap(filename, dtype='float64', mode='w+', shape=shape)


class online_reader():
    """
    a class used to read lazily numpy arrays
    from disk
    """
    def __init__(self,path=None, data_files=None, sort=True, sparse=None, prefix="img", suffix="npy"):
        self.path = path
        self.sparse = sparse
        self.prefix = prefix
        self.suffix = suffix
        self.data_files = data_files
        self.sparse = sparse
        self.dims = None
        if self.data_files is None:
            self.data_files = [os.path.join(self.path, f) for f in os.listdir(self.path)
                               if os.path.isfile(os.path.join(self.path, f)) and f.startswith(self.prefix) and f.endswith(self.suffix)]
        # we sort the files to preserve the order of writing
        if sort:
            self.data_files.sort()
        if self.sparse == "3D" and self.dims is None:
            self.dims = np.load(os.path.join(self.path, "dims.npy")).astype(int)

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, index):
        if isinstance(index, list):
            return self.__getslice__(index[0],index[-1]+1)
        if self.sparse == "2D":
            return load_sparse_matrix(self.data_files[index])
        elif self.sparse == "3D":
            shape = tuple(self.dims[:,index])
            return load_sparse_matrix(self.data_files[index],shape=shape)
        else:
            return np.load(self.data_files[index])

    def __add__(self, other):
        return online_reader(data_files=self.data_files + other.data_files,sort=False,sparse=self.sparse,suffix=self.suffix)

    def __getslice__(self, start, end):
        # return an online reader
        if end-start == 1:
            # if only one image is requested
            return self.__getitem__(0)
        else:
            _online_reader = online_reader(path=self.path, data_files=self.data_files[start:end], prefix=self.prefix, suffix=self.suffix, sparse=self.sparse)
            if self.dims is not None:
                _online_reader.dims = self.dims[:,start:end]
            return _online_reader


class online_writer():

    def __init__(self,path=None, prefix=None, sparse=False, n_files=None):
        self.path = path
        self.prefix = prefix
        self.sparse = sparse
        self.index = 0
        self.n_files = n_files
        # delete the folder if it already exists
        if os.path.exists(self.path):
            shutil.rmtree(self.path)
        os.makedirs(self.path)

    def __len__(self):
        return self.index

    def __getitem__(self, index):
        base = (len(str(self.n_files))-len(str(index)))*'0'
        formated_index = str(base)+str(index)
        return np.load(os.path.join(self.path,self.prefix+formated_index+".npy"))

    def __setitem__(self, index, data):
        base = (len(str(self.n_files))-len(str(index)))*'0'
        formated_index = str(base)+str(index)
        if self.sparse:
            save_sparse_matrix(os.path.join(self.path, self.prefix+formated_index),data)
        else:
            np.save(os.path.join(self.path, self.prefix+formated_index+".npy"),data)

    def append(self, data):
        self.__setitem__(self.index,data)
        self.index += 1

    def __getslice__(self, start, end):
        # return an online reader
        return online_reader(data_files=self.data_files[start:end], suffix=self.suffix, sparse=self.sparse)


def split_dataset(n_training_samples, n_test_samples, y):

    """
    return train and test set partitions of the dataset so that
    there are <n_training_samples> and <n_test_samples> per class
    """

    n_classes = len(set(y))
    idx = [np.array([]).astype(int),np.array([]).astype(int)]
    offset = 0

    for c in range(n_classes):
        yc = y[y == c]
        n_class_samples = yc.size
        trainset_idx = np.random.choice(n_class_samples, size=n_training_samples[c],replace=False)
        testset_idx = np.array([i for i in range(n_class_samples) if i not in trainset_idx]).astype(int)
        if n_test_samples is not None:
            if len(testset_idx) > n_test_samples[c]:
                used_testset_idx = np.random.choice(len(testset_idx), size=n_test_samples[c],replace=False)
                testset_idx = testset_idx[used_testset_idx]

        idx[0] = np.concatenate((idx[0], (offset + trainset_idx)))
        idx[1] = np.concatenate((idx[1], (offset + testset_idx)))
        offset += n_class_samples
    return idx


def explore_dataset(X, img_shape, n_images=5):
    """show sample images randomly selected from the dataset
       assumes datapoints are in columns"""
    import matplotlib.pyplot as plt
    n_samples = X.shape[1]
    idx = np.random.choice(n_samples, n_images, replace=False)
    for i in range(n_images):
        plt.imshow(X[:,idx[i]].reshape(img_shape),cmap=plt.cm.gray)
        plt.show()


def is_img_file(file_path):
    file_types = [".png", ".jpg", ".jpeg", ".pgm"]
    return np.any([file_type in file_path for file_type in file_types])


def resize_img(img, maxdim):
    """
    resizes the image such that width and height are less than
    maxdim (preserving aspect ratio)
    """
    img_arr = np.array(img)
    imdim = max(img_arr.shape[0], img_arr.shape[1])
    if imdim > maxdim:
        scaler = float(maxdim) / float(imdim)
        new_h = int(round(scaler*img_arr.shape[0]))
        new_w = int(round(scaler*img_arr.shape[1]))
        img = img.resize((new_w, new_h))
        img_arr = np.array(img)
    return img_arr


class online_img_reader():
    """
    similar to online_reader class but for image arrays
    """
    def __init__(self, img_files, maxdim, color):
        self.img_files = img_files
        self.maxdim = maxdim
        self.color = color

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        if isinstance(index,list):
            return self.__getslice__(index[0],index[-1]+1)
        img_path = self.img_files[index]
        img = PIL.Image.open(img_path)
        img_arr = np.array(img)
        if self.maxdim is not None:
            # resize the image such that max(width,height) < maxdim
            img_arr = resize_img(img,self.maxdim)
        img = PIL.Image.fromarray(img_arr)
        if not self.color:
            img = img.convert('L')
            img_arr = np.array(img)
        else:
            if len(np.array(img).shape) == 2:
                # case we want RGB but the original image is in grayscale
                img_arr = np.array(img)
                new_img_arr = np.zeros((img_arr.shape[0],img_arr.shape[1],3))
                new_img_arr[:, :, 0] = img_arr
                new_img_arr[:, :, 1] = img_arr
                new_img_arr[:, :, 2] = img_arr
                img_arr = new_img_arr
        return img_arr

    def __getslice__(self, start, end):
        if end-start == 1:
            # if only one image is requested
            return self.__getitem__(0)
        else:
            return online_img_reader(self.img_files[start:end], self.maxdim, self.color)

"""
class img_reader():

    def __init__(self,imgs):
        self.imgs = imgs

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self,index):
        if isinstance(index,list):
            return self.__getslice__(index[0],index[-1]+1)
        return self.imgs[index]


    def __getslice__(self,start,end):
        if end-start == 1:
            #if only one image is requested
            return self.__getitem__(0)
        else:
            return img_reader(self.imgs[start:end])
"""



"""
#this was taken from http://www.umiacs.umd.edu/~zhuolin/projectlcksvd.html and it uses
#Random Face features for the extendedYaleB face dataset.
#The dataset contains 2.414 frontal face images of 38 persons
#with about 64 images per person/class
#32 samples for training and the rest for testing
class extendedYaleB_RandomFace():

    def __init__(self,mmap=True,name="extendedYaleB_RandomFace"):
        self.name = name
        self.mmap = mmap

    def __call__(self):
        import scipy.io
        file_name = "/media/rewire/New Volume/ML_data/RandomFaces4extendedYaleB/randomfaces4extendedyaleb.mat"
        mat = scipy.io.loadmat(file_name, mdict=None)
        self.img_shape = None
        self.n_samples = mat['labelMat'].shape[1]
        self.y = np.zeros((self.n_samples)).astype(np.uint8)
        for i in xrange(self.n_samples):
            self.y[i] = mat['labelMat'][:,i].nonzero()[0][0]


        #Note that mat['filenameMat'][0,c][0,i]
        #is the filename of the ith sample of the cth class
        self.filenames = []
        n_classes = len(set(self.y))
        for c in range(n_classes):
            fnames = mat['filenameMat'][0,c]
            n_class_samples = fnames.shape[1]
            for i in range(n_class_samples):
                self.filenames.append(fnames[0,i][0])

        X = mat['featureMat'][:]
        if self.mmap:
            X = get_mmap(X)

        return X
"""


class AR_RandomFace():
    """
        This was taken from http://www.umiacs.umd.edu/~zhuolin/projectlcksvd.html and it uses
        Random Face features for each person, we randomly select 20 images for training and the other 6 for testing.
    """

    def __init__(self, mmap=True, name="AR RandomFace"):
        self.name = name
        self.mmap = mmap

    def __call__(self):
        import scipy.io
        file_name = "randomfaces4ar.mat"
        mat = scipy.io.loadmat(file_name, mdict=None)

        self.img_shape = None

        self.n_samples = mat['labelMat'].shape[1]
        self.y = np.zeros((self.n_samples)).astype(np.uint8)

        for i in xrange(self.n_samples):
            self.y[i] = mat['labelMat'][:, i].nonzero()[0][0]


        #Note that mat['filenameMat'][0,c][0,i]
        #is the filename of the ith sample of the cth class
        self.filenames = []
        n_classes = len(set(self.y))
        for c in range(n_classes):
            fnames = mat['filenameMat'][0,c]
            n_class_samples = fnames.shape[1]
            for i in range(n_class_samples):
                self.filenames.append(fnames[0,i][0])


        X = mat['featureMat'][:]
        if self.mmap:
            X = get_mmap(X)

        return X


class img_dataset():

    def __init__(self, data_path, max_n_classes=None, min_class_samples=None,
                img_shape=None, new_shape=None, maxdim=None, mmap=True,
                filenames=False, online=False, only_labels=False, color=False, name="image dataset extractor"):
        self.name = name
        self.data_path = data_path
        self.max_n_classes = max_n_classes
        self.min_class_samples = min_class_samples
        self.img_shape = img_shape
        # the shape of the image to be resized to
        # new_shape[0] is the height and new_shape[1] is the width
        self.new_shape = new_shape
        # used to resize the image such that
        # max(height,width) < self.maxdim while
        # preserving the aspect ratio
        self.maxdim = maxdim
        self.y = None
        self.mmap = mmap
        # if true then we will return only the labels
        # and skip extracting thedatapoints
        self.only_labels = only_labels
        self.filenames = filenames
        self.color = color
        self.online = online

    def find_files(self):

        dataset_dir = self.data_path
        dirs = [name for name in os.listdir(dataset_dir)
                if os.path.isdir(os.path.join(dataset_dir, name))]
        self.label_names = dirs
        n_classes = len(dirs)
        if self.max_n_classes is None:
            self.max_n_classes = n_classes

        imgs = []
        img_count = 0
        cl_counts = []
        n_classes_used = 0

        for i in range(n_classes):

            path = os.path.join(dataset_dir, dirs[i])
            class_imgs = []
            for f in os.listdir(path):
                if os.path.isfile(os.path.join(path, f)) and is_img_file(os.path.join(path, f)):
                    class_imgs.append(os.path.join(path, f))

            imgs.append(class_imgs)
            img_count += len(class_imgs)
            cl_counts.append(len(class_imgs))
            n_classes_used += 1
            if(n_classes_used >= self.max_n_classes):
                break

        print "using", n_classes_used, "classes of the dataset", "with a total of", img_count, "images"
        return imgs, cl_counts

    def get_labels(self):
        img_files, cl_counts = self.find_files()
        n_classes_used = len(cl_counts)
        img_count = np.sum([len(img_files[c]) for c in range(n_classes_used)])
        self.n_samples = img_count
        # flatten the list
        self.img_files = [img_file for c in range(n_classes_used) for img_file in img_files[c]]
        y = np.zeros((self.n_samples)).astype(int)
        img_files = []
        cl_start = 0

        for c in range(n_classes_used):
            y[cl_start:cl_start+cl_counts[c]] = c
            cl_start += cl_counts[c]

        self.y = y
        return y

    def lazy_read(self):
        return online_img_reader(self.img_files, self.maxdim, self.color)

    def __call__(self):

        img_files, cl_counts = self.find_files()
        n_classes_used = len(cl_counts)
        img_count = np.sum([len(img_files[c]) for c in range(n_classes_used)])
        self.n_samples = img_count
        self.img_files = [img_file for c in range(n_classes_used) for img_file in img_files[c]]
        y = np.zeros((self.n_samples)).astype(int)
        cl_start = 0

        for c in range(n_classes_used):
            y[cl_start:cl_start+cl_counts[c]] = c
            cl_start += cl_counts[c]

        self.y = y
        if self.filenames:
            return self.img_files
        if self.online:
            return self.lazy_read()

        if not self.only_labels:

            if self.maxdim is None:
                if self.img_shape != None:
                    n_features = self.img_shape[0] * self.img_shape[1]
                if self.new_shape != None:
                    cv_resize_shape = (self.new_shape[1],self.new_shape[0])
                    n_features = self.new_shape[0] * self.new_shape[1]
                if self.mmap:
                    X = get_empty_mmap((self.n_samples,n_features))
                else:
                    X = np.empty((self.n_samples,n_features))
            else:
                images = []
        else:
            return

        for i in range(img_count):

            sys.stdout.write("\rreading images:%3.2f%%" % ((i / float(img_count))*100))
            sys.stdout.flush()
            img_path = self.img_files[i]
            img = PIL.Image.open(img_path)
            if not self.color:
                img = img.convert('L')
            img_arr = np.array(img)

            if self.maxdim is not None:
                img_arr = resize_img(img,self.maxdim)
                images.append(img_arr)
                continue

            if self.new_shape != None:
                # resize the image to a specified shape
                img_arr = np.asarray(img.resize((cv_resize_shape)))

            X[i, :] = img_arr.ravel()

        sys.stdout.write("\rreading images:%3.2f%%" % (100))
        sys.stdout.flush()
        print ""

        self.y = y
        if not self.only_labels:
            if self.maxdim is None:
                return X.T
            else:
                return images
