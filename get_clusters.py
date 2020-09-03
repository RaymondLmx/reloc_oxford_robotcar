import torch
import h5py
import numpy as np
import dataset
import math
from config.template import CONFIG
from torch.utils.data import DataLoader, SubsetRandomSampler
from descriptor import descriptor
from os.path import join, exists
from sklearn.cluster import KMeans
from trajectory import Trajectory
import faiss

# settings
sample_dataset = '2019-01-10-12-32-52-radar-oxford-10k -map'
query_dataset = '2019-01-10-11-46-21-radar-oxford-10k'

cuda = torch.cuda.is_available()
device = torch.device('cuda' if cuda else 'cpu')

# load dataset
print('===> Loading dataset(s)')

query_set = Trajectory(query_dataset)
sample_set = Trajectory(sample_dataset)

cluster_set = dataset.WholeDataset(CONFIG.DATA_PATH, query_set, sample_set, only_sample=True)

print('====> ! %s loaded, sampled %d' % (sample_dataset, sample_rate))


# build model
model = descriptor(mode='cluster', resume=False)
model = model.to(device)

# cluster
print('===> Calculating descriptors and clusters')

n_descriptors = 50000
n_per_radar = 100
n_radar = math.ceil(n_descriptors / n_per_radar)

sampler = SubsetRandomSampler(np.random.choice(len(cluster_set), n_radar, replace=False))
data_loader = DataLoader(dataset=cluster_set,
                         num_workers=CONFIG.N_WORKERS,
                         batch_size=CONFIG.CACHE_BATCH_SIZE,
                         shuffle=False,
                         pin_memory=cuda,
                         sampler=sampler)

init_cache = join('centroids', 'vgg16_train_' + str(CONFIG.N_CLUSTERS) + '_desc_cen.hdf5')
with h5py.File(init_cache, mode='w') as h5:
    with torch.no_grad():
        model.eval()
        print('====> Extracting Descriptors')
        sample_features = h5.create_dataset("descriptors", [n_descriptors, CONFIG.ENCODER_DIM], dtype=np.float32)

        for iteration, (input, indices) in enumerate(data_loader, 1):
            input = input.to(device)
            radar_descriptors = model.encoder(input).view(input.size(0), CONFIG.ENCODER_DIM, -1).permute(0, 2, 1)

            batch_index = (iteration - 1) * CONFIG.CACHE_BATCH_SIZE * n_per_radar
            for index in range(radar_descriptors.size(0)):
                # sample different location for each radar in batch
                sample = np.random.choice(radar_descriptors.size(1), n_per_radar, replace=False)
                start_index = batch_index + index * n_per_radar
                sample_features[start_index: start_index + n_per_radar, :] = \
                    radar_descriptors[index, sample, :].detach().cpu().numpy()

            if iteration % 50 == 0 or len(data_loader) <= 10:
                print("==> Batch ({}/{})".format(iteration, math.ceil(n_radar / CONFIG.CACHE_BATCH_SIZE)), flush=True)
                del input, radar_descriptors

    print('====> Clustering..')
    kmeans = faiss.Kmeans(CONFIG.ENCODER_DIM, CONFIG.N_CLUSTERS, niter=100, verbose=False)
    kmeans.train(sample_features[...])

    print('====> Storing centroids', kmeans.centroids.shape)
    h5.create_dataset('centroids', data=kmeans.centroids)
    print('====> Done!')


    # sikit-learn
    # print('====> Clustering..')
    # kmeans = KMeans(CONFIG.N_CLUSTERS, n_init=100, verbose=False).fit(sample_features[...])
    #
    # print('====> Storing centroids', kmeans.cluster_centers_.shape)
    # h5.create_dataset('centroids', data=kmeans.cluster_centers_)
    # print('====> Done!')
