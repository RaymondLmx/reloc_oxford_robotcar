import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py
import numpy as np
from os.path import join, exists
from torchvision.models import vgg16
from sklearn.neighbors import NearestNeighbors


class NetVLAD(nn.Module):
    '''
        NetVLAD layer define

        Attributes:
            num_clusters    [int]   : number of clusters
            dim             [int]   : dimension of descriptors
            alpha           [float] : parameter of initialization
            input_normalize [bool]  : if apply L_2 normalization to input
    '''
    def __init__(self, num_clusters=64, dim=512, input_normalize=True, vlad_v2=True):
        super(NetVLAD, self).__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.alpha = 0
        self.vlad_v2 = vlad_v2
        self.input_normalize = input_normalize

        self.conv = nn.Conv2d(dim, num_clusters, kernel_size=(1, 1), bias=vlad_v2)
        self.centroids = nn.Parameter(torch.rand(num_clusters, dim))

    def init_parameters(self, clusters, descriptors):

        if self.vlad_v2 == False:
            cluster_assign = clusters / np.linalg.norm(clusters, axis=1, keepdims=True)
            dots = np.dot(cluster_assign, descriptors.T)
            dots.sort(0)
            dots = dots[::-1, :]  # sort, descending

            self.alpha = (-np.log(0.01) / np.mean(dots[0, :] - dots[1, :])).item()
            self.centroids = nn.Parameter(torch.from_numpy(clusters))
            self.conv.weight = nn.Parameter(torch.from_numpy(self.alpha * cluster_assign).unsqueeze(2).unsqueeze(3))
            self.conv.bias = None

        else:
            knn = NearestNeighbors(n_jobs=-1)
            knn.fit(descriptors)
            del descriptors
            descriptors_square = np.square(knn.kneighbors(clusters, 2)[1])
            del knn
            self.alpha = (-np.log(0.01) / np.mean(descriptors_square[:, 1] - descriptors_square[:, 0])).item()
            self.centroids = nn.Parameter(torch.from_numpy(clusters))
            del clusters, descriptors_square

            self.conv.weight = nn.Parameter(
                (2.0 * self.alpha * self.centroids).unsqueeze(-1).unsqueeze(-1)
            )
            self.conv.bias = nn.Parameter(
                - self.alpha * self.centroids.norm(dim=1)
            )

    def forward(self, x):
        '''
            forward path
        '''
        N, C = x.shape[:2]

        if self.input_normalize:
            x = F.normalize(x, p=2, dim=1)

        # soft assignment
        soft_assign = self.conv(x).view(N, self.num_clusters, -1)
        soft_assign = F.softmax(soft_assign, dim=1)

        x_flatten = x.view(N, C, -1)

        # VLAD core: calculate residuals
        vlad = torch.zeros([N, self.num_clusters, C], dtype=x.dtype, layout=x.layout, device=x.device)
        for C in range(self.num_clusters):  # slower than non-looped, but lower memory usage
            residual = x_flatten.unsqueeze(0).permute(1, 0, 2, 3) - \
                       self.centroids[C:C + 1, :].expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
            residual *= soft_assign[:, C:C + 1, :].unsqueeze(2)
            vlad[:, C:C + 1, :] = residual.sum(dim=-1)

        # intra-normalization
        vlad = F.normalize(vlad, p=2, dim=2)

        # L2 normalization
        vlad = vlad.view(x.size(0), -1)
        vlad = F.normalize(vlad, p=2, dim=1)

        return vlad


class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, input):
        return F.normalize(input, p=2, dim=self.dim)


def descriptor(mode, resume, num_clusters=64, dim=512):
    """
        vgg + NetVlad
    """
    encoder = vgg16(pretrained=True)
    layers = list(encoder.features.children())[:-2]

    # if using pretrained then only train conv5_1, conv5_2, and conv5_3
    for l in layers[:-5]:
        for p in l.parameters():
            p.requires_grad = False

    encoder = nn.Sequential(*layers)
    model = nn.Module()
    model.add_module('encoder', encoder)

    if mode != 'cluster':

        netvlad = NetVLAD()
        if not resume:
            if mode == 'train':
                init_cache = join('centroids', 'vgg16_train_' + str(num_clusters) + '_desc_cen.hdf5')
            else:
                init_cache = join('centroids', 'vgg16_test_' + str(num_clusters) + '_desc_cen.hdf5')

            if not exists(init_cache):
                raise FileNotFoundError('Could not find clusters, please run with --mode=cluster before proceeding')

            with h5py.File(init_cache, mode='r') as h5:
                clusters = h5.get('centroids')[...]
                descriptors = h5.get('descriptors')[...]
                netvlad.init_parameters(clusters, descriptors)
                del clusters, descriptors

        model.add_module('pool', netvlad)

    return model

