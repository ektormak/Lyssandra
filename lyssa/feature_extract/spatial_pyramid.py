import numpy as np
from lyssa.utils import get_mmap
from lyssa.feature_extract.dsift import DsiftExtractor

from lyssa.utils.img import grid_patches
from lyssa.utils import run_parallel


class dsift_extractor():
    def __init__(self, step_size=None, patch_size=None):
        # self.step_size = step_size
        self.patch_size = patch_size
        self.extractor = DsiftExtractor(grid_spacing=step_size, patch_size=patch_size)

    def extract(self, img):
        dsift_patches, pos = self.extractor.process_image(img, positionNormalize=False)
        dsift_patches = dsift_patches.T
        pos = pos.T
        return dsift_patches, pos


class patch_extractor():
    def __init__(self, step_size=None, patch_size=None):
        self.step_size = step_size
        self.patch_size = patch_size

    def extract(self, img):
        patches, loc_h, loc_w = grid_patches(img, patch_size=self.patch_size,
                                             step_size=self.step_size, scale=True, return_loc=True)
        pos = np.array([loc_h, loc_w]).T
        return patches, pos


class sc_spm_extractor():
    def __init__(self, feature_extractor=None, levels=(1, 2, 4),
                 sparse_coder=None, pooling_operator=None, normalizer=None):

        self.feature_extractor = feature_extractor
        self.levels = levels
        self.sparse_coder = sparse_coder
        # self.dictionary = dictionary
        self.pooling_operator = pooling_operator
        self.normalizer = normalizer

    def encode(self, imgs, dictionary):

        psize = self.feature_extractor.patch_size
        n_imgs = len(imgs)
        n_atoms = dictionary.shape[1]
        cells = np.array(self.levels) ** 2
        n_features = np.sum(cells) * n_atoms
        Z = np.zeros((n_features, n_imgs))

        for k in range(n_imgs):

            img = imgs[k]
            desc, pos = self.feature_extractor.extract(img)
            # px,py contain the locations of the top-left pixels
            # cx,cy  -//- of the center pixels of each patch
            py = pos[:, 0]
            px = pos[:, 1]
            cy = py + float(psize) / 2 - 0.5
            cx = px + float(psize) / 2 - 0.5

            # sparsely encode the patch
            coded_patches = self.sparse_coder.encode(desc, dictionary)

            n_atoms = coded_patches.shape[0]
            n_total_cells = np.sum(cells)
            imsize = img.shape

            # pre-allocate
            # i.e (n_total_cells,n_atoms)
            poolpatches = np.zeros((n_total_cells, n_atoms))
            cnt = 0
            # iterate over all the cells in the pyramid
            for (i, lev) in enumerate(self.levels):
                # find width and height
                # the cell in current level
                wunit = float(imsize[1]) / lev
                hunit = float(imsize[0]) / lev
                # Find patch-cell memberships
                binidx = np.floor(cy / hunit) * lev + np.floor(cx / wunit)
                for j in range(cells[i]):
                    # get the patch indices of the patches
                    # in the j-th cell of the i-th layer
                    pidx = np.nonzero(binidx == j)[0]
                    if len(pidx) > 0:
                        # pool and then normalize
                        # all the patches in the same cell
                        poolpatches[cnt, :] = self.pooling_operator(coded_patches[:, pidx])
                        if self.normalizer is not None:
                            poolpatches[cnt, :] = self.normalizer(poolpatches[cnt, :])
                    cnt += 1

            Z[:, k] = poolpatches.flatten()
        return Z


def pyramid_feat_extract(imgs, extractor, D):
    return extractor.encode(imgs, D)


class spatial_pyramid():
    """
    A class used to extract ScSPM features from a dataset
    """

    def __init__(self, mmap=False, workspace=None, metadata=None):
        self.workspace = workspace
        self.metadata = metadata
        self.D = None
        self.mmap = mmap

    def extract(self, imgs, pyramid_feat_extractor=None, save=True, n_jobs=4):

        if self.D is None:
            self.D = self.workspace.load("dict.npy")

        n_imgs = len(imgs)
        levels = (1, 2, 4)
        n_atoms = self.D.shape[1]
        n_features = np.sum(np.array(levels) ** 2) * n_atoms

        Z = run_parallel(func=pyramid_feat_extract, data=imgs, args=(pyramid_feat_extractor, self.D),
                         result_shape=(n_features, n_imgs), n_batches=100, mmap=self.mmap,
                         msg="building ScSPM features", n_jobs=n_jobs)

        if save:
            self.workspace.save("features.npy", Z)
        return Z

    def dict_learn(self, imgs, feature_extractor=None, dict_learner=None):
        if not self.workspace.contains("descriptors.npy"):
            self.descriptors = feature_extractor(imgs)
            self.workspace.save("descriptors.npy", self.descriptors)
        else:
            self.descriptors = self.workspace.load("descriptors.npy")

        if self.mmap:
            self.descriptors = get_mmap(self.descriptors)

        print "descriptors extracted"
        if not self.workspace.contains("dict.npy"):
            dict_learner.fit(self.descriptors)
            self.D = dict_learner.D
            self.workspace.save("dict.npy", self.D)
        else:
            self.D = self.workspace.load("dict.npy")
