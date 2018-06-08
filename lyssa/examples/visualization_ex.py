from __future__ import division
import matplotlib.pyplot as plt

from lyssa.dict_learning import ksvd_coder, online_dictionary_coder
from lyssa.sparse_coding import sparse_encoder
from lyssa.feature_extract.preproc import preproc
from lyssa.utils.img import extract_patches

from lyssa.utils import get_images
from sklearn.datasets import fetch_lfw_people


def whitened_rgb_atoms():

    # a small dataset of images
    imgs = get_images(colored=True)

    # alternatively we could use the lfw dataset
    """
    lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4,color=True)
    faces = lfw_people.data
    n_imgs,h,w,n_channels = lfw_people.images.shape
    imgs = []
    for i in range(n_imgs):
        img = lfw_people.images[i,:,:,:].reshape((h,w,n_channels))
        imgs.append(img)
    """

    patch_shape = (8, 8)
    n_atoms = 100
    n_plot_atoms = 100
    n_nonzero_coefs = 1

    print 'Extracting reference patches...'
    X = extract_patches(imgs, patch_size=patch_shape[0], scale=False, n_patches=int(5e5), mem="low")
    print "number of patches:", X.shape[1]

    wn = preproc("whitening")
    from lyssa.feature_extract.preproc import local_contrast_normalization
    # apply lcn and then whiten the patches
    X = wn(local_contrast_normalization(X))
    # learn the dictionary using Batch Orthognal Matching Pursuit and KSVD
    se = sparse_encoder(algorithm='bomp', params={'n_nonzero_coefs': n_nonzero_coefs}, n_jobs=8)
    kc = ksvd_coder(n_atoms=n_atoms, sparse_coder=se, init_dict="data",
                    max_iter=3, verbose=True, approx=False, n_jobs=8)

    kc.fit(X)
    D = kc.D
    for i in range(n_atoms):
        D[:, i] = (D[:, i] - D[:, i].min()) / float((D[:, i].max() - D[:, i].min()))
    # plot the learned dictionary
    plt.figure(figsize=(4.2, 4))
    for i in range(n_plot_atoms):
        plt.subplot(10, 10, i + 1)
        plt.imshow(D[:, i].reshape((patch_shape[0], patch_shape[1], 3)))
        plt.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0, wspace=0.0, hspace=0.0)
        plt.xticks(())
        plt.yticks(())
    plt.show()


def dictionary_learn_ex():

    patch_shape = (18, 18)
    n_atoms = 225
    n_plot_atoms = 225
    n_nonzero_coefs = 2
    n_jobs = 8
    lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4,color=False)
    n_imgs, h, w = lfw_people.images.shape

    imgs = []
    for i in range(n_imgs):
        img = lfw_people.images[i, :, :].reshape((h, w))
        img /= 255.
        imgs.append(img)

    print 'Extracting reference patches...'
    X = extract_patches(imgs, patch_size=patch_shape[0],scale=False,n_patches=int(1e5),verbose=True,n_jobs=n_jobs)
    print "number of patches:", X.shape[1]

    se = sparse_encoder(algorithm='bomp',params={'n_nonzero_coefs': n_nonzero_coefs}, n_jobs=n_jobs)

    odc = online_dictionary_coder(n_atoms=n_atoms, sparse_coder=se, n_epochs=2,
                                  batch_size=1000, non_neg=False, verbose=True, n_jobs=n_jobs)
    odc.fit(X)
    D = odc.D
    plt.figure(figsize=(4.2, 4))
    for i in range(n_plot_atoms):
        plt.subplot(15, 15, i + 1)
        plt.imshow(D[:, i].reshape(patch_shape), cmap=plt.cm.gray)
        plt.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0, wspace=0.0, hspace=0.0)
        plt.xticks(())
        plt.yticks(())
    plt.show()


if __name__ == '__main__':
    #whitened_rgb_atoms()
    dictionary_learn_ex()
