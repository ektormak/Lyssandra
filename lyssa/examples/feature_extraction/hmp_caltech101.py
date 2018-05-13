from lyssa.dict_learning import online_dictionary_coder
from lyssa.sparse_coding import sparse_encoder
from lyssa.feature_extract import hmp_network
from lyssa.utils.dataset import img_dataset
from lyssa.feature_encoding import feature_encoder
from lyssa.utils import workspace_manager


"""
Example code for feature extraction with Hierarchical Matching Pursuit. See
"Hierarchical Matching Pursuit for Image Classification: Architecture and Fast Algorithms"
of Bo et al for the details of the algorithm.
"""


def one_layer_network():

    batch_size = 1000
    n_epochs = 1
    # non-negative dictionary learning
    dl = online_dictionary_coder(batch_size=batch_size, n_epochs=n_epochs, beta=None, non_neg=True, verbose=True)
    se = sparse_encoder(algorithm='nnomp', params={})
    # define the architecture. Here we also use soft thresholding that sets to zero 90% of the coefficients
    # and put a spatial pyramid with 1,2,4 cells in each layer respectively.
    fe = feature_encoder(algorithm="soft_thresholding", params={'nonzero_percentage': 0.1}, verbose=True)
    param_set = {'n_jobs': 8, 'n_layers': 1, 'n_atoms': [256], 'filter_sizes': [7],
                 'step_sizes': [2], 'pooling_sizes': [None], 'pooling_steps': [None],
                 'n_nonzero_coefs': [2], 'dict_learners': [dl], 'spm_levels': (1,2,4),
                 'sparse_coders': [se], 'feature_encoders': [fe], 'n_training_patches': [int(1e6)], 'color': True}


    workspace = workspace_manager()
    workspace.set_metadata(param_set)
    build_hmp_net(param_set,workspace=workspace)


def two_layer_network():

    batch_size = 1000
    dl = online_dictionary_coder(batch_size=batch_size, beta=None, non_neg=False, verbose=True)
    se = sparse_encoder(algorithm='bomp', params={})
    fe = feature_encoder(algorithm="soft_thresholding", params={'nonzero_percentage': 0.1},verbose=True)

    param_grid = {'n_jobs': 8, 'n_layers': 2, 'n_atoms': [64, 128], 'filter_sizes': [3, 3],
                  'step_sizes': [1, 1], 'pooling_sizes': [2, None], 'pooling_steps': [2, None],
                  'n_nonzero_coefs': [1, 2], 'dict_learners': [dl, dl], 'feature_encoders': [fe, fe],
                  'sparse_coders': [se, se], 'n_training_patches': [int(1e6), int(2e5)]}

    workspace = workspace_manager()
    workspace.set_metadata(param_grid)
    build_hmp_net(param_grid,workspace)


def build_hmp_net(param_set,workspace=None):

    # path to the Caltech101 dataset
    data_path = "Caltech101"
    imd = img_dataset(data_path, maxdim=300, online=True, color=param_set.get("color"))
    imgs = imd()
    y = imd.y
    workspace.save("labels.npy", y)
    param_set["dataset_path"] = data_path

    hmp_net = hmp_network(dict_learners=param_set.get("dict_learners"), n_atoms=param_set.get("n_atoms"),
                          filter_sizes=param_set.get("filter_sizes"), n_nonzero_coefs=param_set.get("n_nonzero_coefs"),
                          step_sizes=param_set.get("step_sizes"), pooling_sizes=param_set.get("pooling_sizes"),
                          spp_pooler = param_set.get('spp_pooler'), pooling_steps=param_set.get("pooling_steps"),
                          spm_levels = param_set.get("spm_levels"), rebuild_spp = param_set.get('rebuild_spp'),
                          sparse_coders=param_set.get("sparse_coders"), feature_encoders=param_set.get("feature_encoders"),
                          workspace=workspace,pretrained_dicts = param_set.get('pretrained_dicts'),
                          n_training_patches=param_set.get("n_training_patches"), imgs=imgs,mmap=False,n_jobs=param_set.get("n_jobs"))

    hmp_net.build_hierarchy()
    print "finished!"


if __name__ == '__main__':

    one_layer_network()
    #two_layer_network()
