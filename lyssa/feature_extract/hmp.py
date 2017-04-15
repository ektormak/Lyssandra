import numpy as np
from lyssa.utils.img import grid_patches_loc, extract_patches, compute_n_patches

from lyssa.utils.dataset import online_reader, save_sparse_matrix
from lyssa.utils import run_parallel
import sys
import os
import multiprocessing
from itertools import product
from functools import partial
from scipy import sparse


def save_feature_map_proc(fmap_path, Z):
    save_sparse_matrix(fmap_path, sparse.csc_matrix(Z))


def pool_proc(feature_maps, feature_map_dims, idxs, output_img_path, spp=False, conv=False, pooling_size=None,
              normalizer=None, spp_pooler=None, pooling_step=None, levels=(1, 2, 4)):
    n_imgs = len(feature_maps)
    n_features = feature_maps[0].shape[0]

    if spp:
        n_cells = np.array(levels) ** 2
        n_total_cells = np.sum(n_cells)
        # pre-allocate
        Z_final = np.zeros((n_total_cells * n_features, n_imgs))

    if conv and pooling_size is not None:
        offset = np.array(list(product(np.arange(pooling_size), np.arange(pooling_size))))

    for i in range(n_imgs):

        feature_map = feature_maps[i]
        ph = feature_map_dims[0, i]
        pw = feature_map_dims[1, i]

        if spp:

            loc_h = np.arange(ph)
            loc_w = np.arange(pw)
            locs = np.array(list(product(loc_h, loc_w)))
            loc_h = locs[:, 0]
            loc_w = locs[:, 1]

            # pre-allocate
            # i.e (n_total_cells,n_atoms)
            poolpatches = np.zeros((n_total_cells, n_features))
            cnt = 0
            # iterate over each cell and level in the pyramid
            for (c, lev) in enumerate(levels):
                # find the cell width and height
                wunit = float(pw) / lev
                hunit = float(ph) / lev
                # find patch-cell memberships
                binidx = np.floor(loc_h / hunit) * lev + np.floor(loc_w / wunit)
                for k in range(n_cells[c]):
                    idx = np.nonzero(binidx == k)[0]
                    hidx = loc_h[idx]
                    widx = loc_w[idx]
                    if len(idx) > 0:
                        pmap = feature_map[:, idx]
                        poolpatches[cnt, :] = spp_pooler(pmap)
                    cnt += 1

            Z_final[:, i] = poolpatches.flatten()

        if conv:

            if pooling_size is not None:

                pool_loc_h, pool_loc_w = grid_patches_loc(ph, pw, pooling_size, pooling_step)
                pool_h, pool_w = compute_n_patches(ph, pw, pooling_size, pooling_step, padding=False)
                pooled_img = np.zeros((pool_h, pool_w, n_features))
                # indices of the pooling groups
                # in the pooled feature map
                pp_idx = np.array(list(product(np.arange(pool_h), np.arange(pool_w))))
                n_pooling_groups = len(pool_loc_h)
                # take the absolute value of the feature map
                feature_map = np.abs(feature_map)

                for pg in range(n_pooling_groups):

                    idx = []
                    for o in range(len(offset)):
                        oh = offset[o, 0]
                        ow = offset[o, 1]

                        hidx = ((pool_loc_h[pg] + oh) * pw)
                        widx = (pool_loc_w[pg] + ow)
                        idx.append(hidx + widx)

                    pmap = feature_map[:, idx]
                    try:
                        pmap = np.max(pmap, axis=1)
                        pooled_img[pp_idx[pg, 0], pp_idx[pg, 1], :] = pmap
                    except:
                        import pdb;
                        pdb.set_trace()

                save_sparse_matrix(os.path.join(output_img_path, "img" + str(idxs[i]) + ".npy"),
                                   sparse.csc_matrix(pooled_img.reshape((pool_h * pool_w, n_features))))
                img_dims = pool_h, pool_w, n_features
            else:
                # convolution without pooling
                pooled_img = feature_map.reshape((ph, pw, n_features))
                # reshape to 2D and save it
                save_sparse_matrix(os.path.join(output_img_path, "img" + str(idxs[i]) + ".npy"),
                                   sparse.csc_matrix(pooled_img.reshape((ph * pw, n_features))))
                # save the dimensions
                img_dims = ph, pw, n_features

            # save the image to be used as input to the next layer
            np.save(os.path.join(output_img_path, "img_dim" + str(idxs[i]) + ".npy"), img_dims)

    if spp:
        return Z_final


def compute_training_set_mem(n_training_patches=1e6, patch_size=None, patch_depth=None):
    float_size = 8
    n_gbs = (n_training_patches * (patch_size ** 2) * patch_depth * float_size) / float(1024 ** 3)
    return n_gbs


class hmp_network():
    """
    a class used to build Hierarchical Matching Pursuit Networks. See
    "Hierarchical Matching Pursuit for Image Classification: Architecture and Fast Algorithms"
    of Bo et al for the details of the algorithm.
    """

    def __init__(self, dict_learners=None, n_atoms=None, sparse_coders=None, feature_encoders=None, filter_sizes=None,
                 step_sizes=None, n_nonzero_coefs=None, pooling_sizes=None, pooling_steps=None, pre_procs=None,
                 spm_levels=(1, 2, 4), rebuild_spp=False, spm_normalizer=None, spp_pooler=None, n_training_patches=None,
                 imgs=None,
                 pretrained_dicts=None, workspace=None, mmap=False, verbose=False, n_jobs=1):

        self.filter_sizes = filter_sizes
        self.n_nonzero_coefs = n_nonzero_coefs
        self.n_layers = len(filter_sizes)
        self.step_sizes = step_sizes
        self.n_atoms = n_atoms
        self.dict_learners = dict_learners
        self.sparse_coders = sparse_coders
        self.feature_encoders = feature_encoders
        self.verbose = verbose
        self.rebuild_spp = rebuild_spp
        if pretrained_dicts is None:
            self.pretrained_dicts = [None for l in range(self.n_layers)]
        else:
            self.pretrained_dicts = pretrained_dicts

        if spp_pooler is None:
            from .pooling import sc_max_pooling
            self.spp_pooler = sc_max_pooling()
        else:
            self.spp_pooler = spp_pooler

        for l in range(self.n_layers):
            self.feature_encoders[l].n_jobs = n_jobs

            self.sparse_coders[l].n_jobs = n_jobs
            self.sparse_coders[l].params['n_nonzero_coefs'] = self.n_nonzero_coefs[l]
            self.sparse_coders[l].verbose = verbose

        self.pooling_sizes = pooling_sizes
        self.pooling_steps = pooling_steps
        self.pre_procs = pre_procs
        self.spm_levels = spm_levels
        if spm_normalizer is None:
            # from .preproc import l2_normalizer
            # self.spm_normalizer = l2_normalizer()
            self.spm_normalizer = None
        # number of training patches to use for
        # dictionary learning in each layer
        self.n_training_patches = n_training_patches
        # after building each layer, self.imgs[layer]
        # will be a list of 2D images of the layer
        self.imgs = [None for l in range(self.n_layers + 1)]
        self.feature_maps = [None for l in range(self.n_layers)]
        # the first layer contains the raw image data
        self.imgs[0] = imgs
        self.workspace = workspace
        self.mmap = mmap
        self.n_jobs = n_jobs
        print self.show_details()

    def show_details(self):
        im_heights = []
        im_widths = []
        n_imgs = len(self.imgs[0])
        for i in range(n_imgs):
            h, w = self.imgs[0][i].shape[:2]
            im_heights.append(h)
            im_widths.append(w)
        avg_im_height = np.mean(im_heights)
        avg_im_width = np.mean(im_widths)
        print "average image height: {0}".format(avg_im_height)
        print "average image width: {0}".format(avg_im_width)
        print "------------------------------------"
        for l in range(self.n_layers):
            if l == 0:
                patch_depth = 3
            else:
                patch_depth = self.n_atoms[l - 1]

            avg_height_after_conv, avg_width_after_conv = compute_n_patches(avg_im_height, avg_im_width,
                                                                            self.filter_sizes[l], self.step_sizes[l],
                                                                            padding=False)
            print "average image height after convolutional layer: {0}".format(avg_height_after_conv)
            print "average image width after convolutional layer: {0}".format(avg_width_after_conv)
            if self.pooling_sizes[l] is not None:
                avg_height_after_pooling, avg_width_after_pooling = compute_n_patches(avg_height_after_conv,
                                                                                      avg_width_after_conv,
                                                                                      self.pooling_sizes[l],
                                                                                      self.pooling_steps[l],
                                                                                      padding=False)
                print "average image height after pooling layer: {0}".format(avg_height_after_pooling)
                print "average image width after pooling layer: {0}".format(avg_width_after_pooling)
                avg_im_height = avg_height_after_pooling
                avg_im_width = avg_width_after_pooling
            else:
                avg_im_height = avg_height_after_conv
                avg_im_width = avg_width_after_conv
            size_in_gb = compute_training_set_mem(n_training_patches=self.n_training_patches[l],
                                                  patch_size=self.filter_sizes[l], patch_depth=patch_depth)
            print "memory size for dictionary learning:", size_in_gb
            print "-------------------"
        print "------------------------------------"

    def learn_dict(self, layer):

        n_imgs = len(self.imgs[layer])
        is_grayscale = len(self.imgs[layer][0].shape) == 2
        n_patches = self.n_training_patches[layer]
        patch_size = self.filter_sizes[layer]
        print "---------------------------------------"
        print "learning the dictionary for layer", layer
        print "number of patches:", n_patches
        print "number of atoms:", self.n_atoms[layer]

        X = extract_patches(self.imgs[layer], n_patches=n_patches, patch_size=patch_size,
                            mmap=False, mem="low", n_jobs=self.n_jobs, verbose=True, scale=False)

        if layer == 0:
            # scale to [0,1]
            X /= 255.
            if self.pre_procs is not None:
                for p in range(len(self.pre_procs)):
                    X = self.pre_procs[p](X)
        elif layer > 0:
            # do contrast normalization
            pass

        self.dict_learners[layer].sparse_coder = self.sparse_coders[layer]
        self.dict_learners[layer].n_atoms = self.n_atoms[layer]
        self.dict_learners[layer].fit(X)
        D = self.dict_learners[layer].D
        self.workspace.save(os.path.join("layer" + str(layer), "dict.npy"), D)
        return D

    def build_feature_maps(self, layer):

        if (not self.workspace.contains(os.path.join("layer" + str(layer), "dict.npy"))) and self.pretrained_dicts[
            layer] is None:
            D = self.learn_dict(layer)
        elif self.pretrained_dicts[layer] is not None:
            D = self.pretrained_dicts[layer]
            self.workspace.save(os.path.join("layer" + str(layer), "dict.npy"), D)
        else:
            D = self.workspace.load(os.path.join("layer" + str(layer), "dict.npy"))

        if layer > 0:
            input_img_path = "layer" + str(layer) + "/imgs"
            self.imgs[layer] = online_reader(path=os.path.join(self.workspace.base_path, input_img_path), sparse="3D",
                                             prefix="img", suffix="npz")
        feature_maps_path = "layer" + str(layer) + "/feature_maps"
        os.path.join("layer" + str(layer), "feature_map_dims.npy")

        if not os.path.exists(
                os.path.join(self.workspace.base_path, os.path.join("layer" + str(layer), "feature_map_dims.npy"))):
            if not os.path.exists(os.path.join(self.workspace.base_path, feature_maps_path)):
                os.makedirs(os.path.join(self.workspace.base_path, feature_maps_path))
            n_imgs = len(self.imgs[layer])
            is_grayscale = len(self.imgs[layer][0].shape) == 2
            patch_size = self.filter_sizes[layer]
            idxs = [(len(str(n_imgs)) - len(str(i))) * '0' + str(i) for i in range(n_imgs)]
            if layer == 0:
                batch_size = 100
            if layer > 0:
                batch_size = 20
            from lyssa.utils import gen_batches
            batch_idxs = gen_batches(n_imgs, batch_size=batch_size)
            n_batches = len(batch_idxs)
            feature_map_dims = np.zeros((n_imgs, 2))
            for i in range(n_batches):
                sys.stdout.write("\r" + "building feature maps" + ":%3.2f%%" % ((i / float(n_batches)) * 100))
                sys.stdout.flush()
                X, patch_sizes = extract_patches(self.imgs[layer][batch_idxs[i]],
                                                 step_size=self.step_sizes[layer], patch_size=self.filter_sizes[layer],
                                                 mmap=self.mmap, n_jobs=self.n_jobs, scale=False, verbose=False,
                                                 mem="high")

                if layer == 0:
                    # scale to [0,1]
                    X /= 255.
                elif layer > 0:
                    # do contrast normalization
                    pass

                Z = self.feature_encoders[layer].encode(X, D)
                start = 0
                for l, j in enumerate(batch_idxs[i]):
                    img = self.imgs[layer][j]

                    h, w = img.shape[:2]
                    ph, pw = compute_n_patches(h, w, self.filter_sizes[layer], self.step_sizes[layer], padding=False)
                    end = start + patch_sizes[l]
                    fmap_path = os.path.join(self.workspace.base_path, os.path.join(feature_maps_path, "img" + idxs[j]))
                    multiprocessing.Process(target=save_feature_map_proc, args=(fmap_path, Z[:, start:end])).start()
                    start += patch_sizes[l]
                    feature_map_dims[j, 0] = ph
                    feature_map_dims[j, 1] = pw

        self.workspace.save(os.path.join("layer" + str(layer), "feature_map_dims.npy"), feature_map_dims.astype(int))
        self.feature_maps[layer] = self.workspace.load(feature_maps_path, sparse="2D", online=True)

    def build_layer(self, layer=0):

        feature_maps_path = "layer" + str(layer) + "/feature_maps"
        output_img_path = os.path.join(self.workspace.base_path, "layer" + str(layer + 1) + "/imgs")
        if not os.path.exists(
                os.path.join(self.workspace.base_path, os.path.join("layer" + str(layer), "feature_map_dims.npy"))):
            self.build_feature_maps(layer)

        self.feature_maps[layer] = self.workspace.load(feature_maps_path, sparse="2D", online=True)
        levels = self.spm_levels
        features_path = os.path.join("layer" + str(layer), "features.npy")
        feature_map_dims = self.workspace.load(os.path.join("layer" + str(layer), "feature_map_dims.npy"), online=False)
        feature_map_dims = feature_map_dims.astype(int)
        n_imgs = len(self.feature_maps[layer])
        # make the formated indices
        idxs = [(len(str(n_imgs)) - len(str(i))) * '0' + str(i) for i in range(n_imgs)]
        spp = False
        conv = False
        pooling_step = None
        pooling_size = None
        if (not self.workspace.contains(features_path)) or self.rebuild_spp:
            n_cells = np.array(levels) ** 2
            n_total_cells = np.sum(n_cells)
            n_features = self.n_atoms[layer]
            # pre-allocate
            Z_final = np.zeros((n_total_cells * n_features, n_imgs))
            patch_size = self.filter_sizes[layer]

            spp = True

        if not os.path.exists(output_img_path):

            if layer < self.n_layers - 1:
                conv = True
                pooling_step = self.pooling_steps[layer]
                pooling_size = self.pooling_sizes[layer]
                os.makedirs(output_img_path)
            else:
                conv = False
                output_img_path = None

        if spp:
            print "building SPP layer on top of layer{0}".format(layer)
        if conv:
            print "building CONV layer on top of layer{0}".format(layer)

        if spp or conv:
            # if the layer is not done yet
            func = partial(pool_proc, pooling_size=pooling_size, spp_pooler=self.spp_pooler,
                           pooling_step=pooling_step, spp=spp, conv=conv, normalizer=self.spm_normalizer,
                           levels=self.spm_levels)
            n_batches = 100
            msg = "building next layer"
            data = self.feature_maps[layer]
            if spp:

                Z_final = run_parallel(func=func, data=data, args=[output_img_path],
                                       batched_args=[feature_map_dims.T, idxs],
                                       result_shape=(n_total_cells * n_features, n_imgs), n_batches=n_batches,
                                       mmap=self.mmap, msg=msg, n_jobs=self.n_jobs)
                self.workspace.save(features_path, Z_final)
            else:

                run_parallel(func=func, data=data, args=[output_img_path], batched_args=[feature_map_dims.T, idxs],
                             result_shape=None, n_batches=n_batches,
                             mmap=self.mmap, msg=msg, n_jobs=self.n_jobs)

            if conv:
                img_dims = np.zeros((3, n_imgs)).astype(int)
                img_dim_files = [os.path.join(output_img_path, f) for f in os.listdir(output_img_path)
                                 if os.path.isfile(os.path.join(output_img_path, f)) and f.startswith("img_dim")]
                img_dim_files.sort()
                for i, img_dim_file in enumerate(img_dim_files):
                    dims = np.load(img_dim_file)
                    img_dims[:, i] = dims
                    os.remove(img_dim_file)
                self.workspace.save(os.path.join(output_img_path, "dims.npy"), img_dims)

        if layer < self.n_layers - 1:
            self.imgs[layer + 1] = online_reader(path=output_img_path, sparse="3D", prefix="img", suffix="npz")

    def build_hierarchy(self, layer=0):
        # start building from the
        # specified <layer> and above
        if layer > 0:
            img_path = os.path.join("layer" + str(layer), "imgs")
            self.imgs[layer] = self.workspace.load(img_path, online=True)

        for l in range(layer, self.n_layers):
            self.build_layer(l)

    """
    def visualize_activations(self,input_img):

        #buid the feature maps of the input image
        feature_maps = []
        step = 1
        #save_path = os.path.join(self.save_dir,"activations")
        #if os.path.exists(save_path):
        #		shutil.rmtree(save_path)
        #os.makedirs(save_path)
        img = input_img
        for layer in range(self.n_layers):
            n_atoms = self.n_atoms[layer]
            patch_size = self.patch_sizes[layer]
            is_grayscale = len(img.shape) == 2
            #scale to [0,1]
            if layer == 0 and is_grayscale:
                img = np.array(img,dtype=np.float) / 255.
            else:
                #maybe do contrast normalization
                img = np.array(img,dtype=np.float)
            h,w = img.shape[:2]
            #the number of patches in the image
            #is ph*pw
            ph = h - patch_size + 1
            pw = w - patch_size + 1
            loc_h,loc_w = grid_patches_loc(img, patch_size, step)
            n_patches = len(loc_h)
            if layer == 0:
                X = np.zeros((patch_size**2,n_patches))
            else:
                #since each "pixel" here contains self.n_atoms[layer-1] entries
                X = np.zeros(( (patch_size**2)*self.n_atoms[layer-1],n_patches))
            #print "extracting patches"
            for j in range(n_patches):
                #for the first layer
                if layer == 0 and is_grayscale:
                    patch = img[loc_h[j]:loc_h[j]+patch_size,loc_w[j]:loc_w[j]+patch_size]
                elif layer>0:

                    patch = img[loc_h[j]:loc_h[j]+patch_size,loc_w[j]:loc_w[j]+patch_size,:]
                    #contrast normalize this patch


                #maybe this is wrong for layer>0
                patch = patch.ravel()
                X[:,j] = patch


            Z = self.sparse_coders(X,self.dicts[layer])


            encoded_img = Z.reshape((ph,pw,n_atoms))
            #patched_img = X.reshape((ph,pw,X.shape[0]))

            #the feature maps of this layer
            feature_maps.append([encoded_img[:,:,k] for k in range(n_atoms)])

            #layer_path = os.path.join(save_dir,"layer")
            #img_path = os.path.join(layer_path+str(layer+1),"imgs")
            #patched_path = os.path.join(layer_path+str(layer),"patched_imgs")
            #encoded_path = os.path.join(layer_path+str(layer),"encoded_imgs")

            #we wont store this for the first layer

            if pooling_size is not None:
                #pool_h * pool_w are the total number of pooling
                #regions in the image
                pool_h = int(floor(ph/float(pooling_size)))
                pool_w = int(floor(pw/float(pooling_size)))

                pooled_img = np.zeros((pool_h,pool_w,n_atoms))
                for sh in range(pool_h):
                    for sw in range(pool_w):
                        #determine the patches that are in the (sh,sw) pooling group
                        ph_idx = pooling_size * sh + np.arange(pooling_size)
                        pw_idx = pooling_size * sw + np.arange(pooling_size)
                        abs_codes = np.abs(encoded_img[ph_idx,:][:,pw_idx]).reshape(pooling_size**2,
                                                                            n_atoms)
                        max_codes = np.max(abs_codes,axis=0)
                        pooled_img[sh,sw] = max_codes

                #save the image to be used as input to the next layer
                #np.save(os.path.join(img_path,"img"+str(index)+".npy"),pooled_img)
                img = pooled_img


        def get_activation(l,atom_idx):

            return feature_maps[l][atom_idx]

        return get_activation
    """
