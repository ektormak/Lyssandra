from lyssa.utils.dataset import img_dataset
from lyssa.utils.workspace import workspace_manager
from lyssa.sparse_coding import sparse_encoder
from lyssa.dict_learning import online_dictionary_coder
from lyssa.feature_extract.spatial_pyramid import dsift_extractor, spatial_pyramid, sc_spm_extractor
from lyssa.feature_extract.pooling import sc_max_pooling
from lyssa.feature_extract.preproc import l2_normalizer

"""
Feature extraction using Spatial Pyramid Matching and Sparse Coding on top
of SIFT features. See the paper "Linear Spatial Pyramid Matching Using Sparse Coding
for Image Classification" for the details of the algorithm.
"""


def ScSPM_caltech101_l1():
    # path to the Caltech101 dataset
    data_path = "Caltech101/"
    imd = img_dataset(data_path, maxdim=300, online=True, color=param_set.get("color"))
    imgs = imd()
    y = imd.y

    # define the architecture
    n_atoms = 1024
    patch_shape = (16, 16)
    step_size = 6
    n_descriptors = int(1e6)
    # the l1 penalty parameter
    _lambda = 0.15
    normalizer = l2_normalizer
    metadata = {'name': "ScSPM_caltech101",
                'desc': "ScSPM using sparse coding on top of SIFT features",
                'n_atoms': n_atoms,
                'lambda': _lambda,
                'patch_shape': patch_shape,
                'step_size': step_size,
                'n_descriptors': n_descriptors,
                'pooling': 'max pooling',
                'normalization': 'l2'}

    wm = workspace_manager(metadata=metadata)
    wm.set_metadata(metadata)
    wm.save("labels.npy", y)
    sp = spatial_pyramid("ScSPM", workspace=wm, mmap=False)
    n_jobs = 8
    # the sift extractor of the
    # initial phase of dictionary learning
    feature_extractor = sift_extractor(n_descriptors=n_descriptors, patch_shape=patch_shape, mmap=False, n_jobs=n_jobs)
    se = sparse_encoder(algorithm='lasso', params={'lambda': n_nonzero_coefs}, n_jobs=n_jobs)
    odc = online_dictionary_coder(n_atoms=n_atoms, sparse_coder=se,
                                  batch_size=int(1e3), verbose=True, mmap=False, n_jobs=n_jobs)

    # learn the dictionary
    sp.dict_learn(imgs, feature_extractor=feature_extractor, dict_learner=odc)
    se.n_jobs = 1
    # extract ScSPM features
    sc_spm = sc_spm_extractor(feature_extractor=dsift_extractor(step_size=step_size, patch_size=patch_shape[0]),
                              levels=(1, 2, 4), sparse_coder=se, pooling_operator=sc_max_pooling(),
                              normalizer=normalizer)

    sp.extract(imgs, pyramid_feat_extractor=sc_spm, n_jobs=n_jobs)


if __name__ == "__main__":
    ScSPM_caltech101_l1()
