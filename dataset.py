import os
import torch
import numpy as np
import h5py
import torch.utils.data as data
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from tool.radar import load_radar
from sklearn.neighbors import NearestNeighbors
from torchvision import transforms
from config.template import CONFIG
from trajectory import Trajectory

# settings
root = CONFIG.DATA_PATH
sample_rate = 0


def get_whole_train_set(only_sample=False):

    query_set = Trajectory('2019-01-10-11-46-21-radar-oxford-10k')
    sample_set = Trajectory('2019-01-10-12-32-52-radar-oxford-10k -map')

    if sample_rate > 1:
        query_set.downsample(sample_rate)
        sample_set.downsample(sample_rate)

    return WholeDataset(root, query_set, sample_set, only_sample=only_sample)


def get_triplet_train_set():

    query_set = Trajectory('2019-01-10-11-46-21-radar-oxford-10k')
    sample_set = Trajectory('2019-01-10-12-32-52-radar-oxford-10k -map')

    if sample_rate > 1:
        query_set.downsample(sample_rate)
        sample_set.downsample(sample_rate)

    return TripletDataset(root, query_set, sample_set)


def get_whole_val_set():

    query_set = Trajectory('2019-01-10-11-46-21-radar-oxford-10k')
    sample_set = Trajectory('2019-01-10-12-32-52-radar-oxford-10k -map')

    query_set.intercept(7000, 8000)

    return WholeDataset(root, query_set, sample_set, only_sample=False)


def get_whole_test_set():

    query_set = Trajectory('2019-01-10-14-50-05-radar-oxford-10k')
    sample_set = Trajectory('2019-01-10-12-32-52-radar-oxford-10k -map')

    query_set.intercept(5000, 6000)

    return WholeDataset(root, query_set, sample_set, only_sample=False)


def radar_to_tensor(radar):
    """
        transform radar to tensor (3, 400, 450)
    """

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((400, 450)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                      std=[0.229, 0.224, 0.225]),
    ])

    radar_tensor = torch.from_numpy(np.squeeze(radar))
    radar_tensor = radar_tensor.unsqueeze(0)
    radar_tensor = torch.cat((radar_tensor, radar_tensor, radar_tensor), 0)
    radar_tensor = transform(radar_tensor)

    return radar_tensor


class WholeDataset(Dataset):
    """
        load radar data
    """
    def __init__(self, root, query_set, sample_set, only_sample):
        super().__init__()

        self.root = root
        self.query_set = query_set
        self.sample_set = sample_set

        # list of samples path is in the front
        queries_path = os.path.join(root, query_set.dataset)
        samples_path = os.path.join(root, sample_set.dataset)
        self.radars_path = [os.path.join(samples_path, 'radar', '%d.png' % r) for r in self.sample_set.time_stamps]
        if not only_sample:
            self.radars_path += [os.path.join(queries_path, 'radar', '%d.png' % r) for r in self.query_set.time_stamps]

        self.positives = None
        self.distances = None

    def __getitem__(self, index):

        radar_path = self.radars_path[index]
        # if not os.path.isfile(radar_path):
        #     raise IOError("Could not find radar scan" + radar_path)
        # radar_times_tamps, azimuths, valid, fft_data, radar_resolution
        _, _, _, radar, _ = load_radar(radar_path)

        # transform
        radar_tensor = radar_to_tensor(radar)

        return radar_tensor, index

    def __len__(self):
        return len(self.radars_path)

    def get_positives(self):
        # positives for evaluation are those within trivial threshold range
        # fit NN to find them, search by radius
        if self.positives is None:
            knn = NearestNeighbors(n_jobs=-1)
            knn.fit(self.sample_set.ground_truth)

            self.distances, self.positives = knn.radius_neighbors(self.query_set.ground_truth,
                                                                  radius=CONFIG.NONTRIV_POS_THREADS)

        return self.positives


def collate_fn(batch):
    """Creates mini-batch tensors from the list of tuples (query, positive, negatives).

    Args:
        data: list of tuple (query, positive, negatives).
            - query: torch tensor of shape (3, h, w).
            - positive: torch tensor of shape (3, h, w).
            - negative: torch tensor of shape (n, 3, h, w).
    Returns:
        query: torch tensor of shape (batch_size, 3, h, w).
        positive: torch tensor of shape (batch_size, 3, h, w).
        negatives: torch tensor of shape (batch_size, n, 3, h, w).
    """

    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0: return None, None, None, None, None

    query, positive, negatives, indices = zip(*batch)

    query = data.dataloader.default_collate(query)
    positive = data.dataloader.default_collate(positive)
    negCounts = data.dataloader.default_collate([x.shape[0] for x in negatives])
    negatives = torch.cat(negatives, 0)
    import itertools
    indices = list(itertools.chain(*indices))

    return query, positive, negatives, negCounts, indices


class TripletDataset(Dataset):
    """
        query dataset as triplet structure
    """
    def __init__(self, root, query_set, sample_set, margin=0.1, n_neg_sample=1000, n_neg=10):
        super().__init__()

        # load queries and samples
        self.root = root
        self.query_set = query_set
        self.sample_set = sample_set
        self.margin = margin
        self.n_neg_sample = n_neg_sample
        self.n_neg = n_neg

        # find potential positives
        knn = NearestNeighbors(n_jobs=-1)
        knn.fit(self.sample_set.ground_truth)

        self.nontrivial_positives = list(knn.radius_neighbors(self.query_set.ground_truth,
                                                              radius=CONFIG.NONTRIV_POS_THREADS,
                                                              return_distance=False))
        for i, ntpos in enumerate(self.nontrivial_positives):
            self.nontrivial_positives[i] = np.sort(ntpos)

        self.queries_index = np.where(np.array([len(x) for x in self.nontrivial_positives])>0)[0]

        # find potential negatives
        potential_positives = knn.radius_neighbors(self.query_set.ground_truth,
                                                   radius=CONFIG.POS_THREADS,
                                                   return_distance=False)
        self.potential_negatives = []
        for pos in potential_positives:
            self.potential_negatives.append(np.setdiff1d(np.arange(self.sample_set.num), pos, assume_unique=True))

        self.cache = None
        self.neg_cache = [np.empty((0,)) for _ in range(self.query_set.num)]

    def __getitem__(self, index):
        index = self.queries_index[index]
        with h5py.File(self.cache, mode='r') as h5:
            h5_features = h5.get('features')

            queries_feature = h5_features[index + self.sample_set.num]

            # positive
            pos_feature = h5_features[self.nontrivial_positives[index].tolist()]
            knn = NearestNeighbors(n_jobs=1)
            knn.fit(pos_feature)
            pos_dist, pos_nn_index = knn.kneighbors(queries_feature.reshape(1, -1), 1)
            pos_dist = pos_dist.item()
            pos_index = self.nontrivial_positives[index][pos_nn_index[0]].item()

            # negative sample
            neg_sample = np.random.choice(self.potential_negatives[index], self.n_neg_sample)
            neg_sample = np.unique(np.concatenate([self.neg_cache[index], neg_sample]))

            neg_feature = h5_features[neg_sample.tolist()]
            knn.fit(neg_feature)

            neg_dist, neg_nn_index = knn.kneighbors(queries_feature.reshape(1, -1), self.n_neg*10)
            neg_dist = neg_dist.reshape(-1)
            neg_nn_index = neg_nn_index.reshape(-1)

            violating_neg = neg_dist < pos_dist + self.margin**0.5

            if np.sum(violating_neg) < 1:
                return None

            neg_nn_index = neg_nn_index[violating_neg][:self.n_neg]
            neg_indices = neg_sample[neg_nn_index].astype(np.int32)
            self.neg_cache[index] = neg_indices

        # query radar
        radar_path = os.path.join(self.root,
                                  self.query_set.dataset,
                                  'radar', '%d.png' % self.query_set.time_stamps[index])
        # if not os.path.isfile(radar_path):
        #     raise IOError("Could not find radar scan" + radar_path)
        _, _, _, query, _ = load_radar(radar_path)
        query = radar_to_tensor(query)

        # positive radar
        radar_path = os.path.join(self.root,
                                  self.sample_set.dataset,
                                  'radar', '%d.png' % self.sample_set.time_stamps[pos_index])
        # if not os.path.isfile(radar_path):
        #     raise IOError("Could not find radar scan" + radar_path)
        _, _, _, positive, _ = load_radar(radar_path)
        positive = radar_to_tensor(positive)

        # negative radars
        negatives = []
        for neg_index in neg_indices:
            radar_path = os.path.join(self.root,
                                      self.sample_set.dataset,
                                      'radar', '%d.png' % self.sample_set.time_stamps[neg_index])
            # if not os.path.isfile(radar_path):
            #     raise IOError("Could not find radar scan" + radar_path)
            _, _, _, negative, _ = load_radar(radar_path)
            negative = radar_to_tensor(negative)
            negatives.append(negative)

        negatives = torch.stack(negatives, 0)

        # plot
        # self.plot_batch(index, pos_index, neg_indices)

        return query, positive, negatives, [index, pos_index] + neg_indices.tolist()

    def __len__(self):
        return len(self.queries_index)

    def plot_batch(self, query_index, pos_index, neg_indices):
        """
            plot batch on trajectory to verify its
        """
        query_gt = self.query_set.ground_truth[query_index]
        pos_gt = self.sample_set.ground_truth[pos_index]
        neg_gt = []
        for ni in neg_indices:
            neg_gt.append(self.sample_set.ground_truth[ni])
        neg_gt = np.array(neg_gt)

        plt.figure(figsize=(10, 10))

        plt.title('trajectory', fontsize=15)
        plt.xlabel('x', fontsize=14)
        plt.ylabel('y', fontsize=14)
        plt.xlim(-200, 1200)
        plt.ylim(-500, 900)
        plt.scatter(self.sample_set.ground_truth[:, 0], self.sample_set.ground_truth[:, 1], s=0.5, c='y')
        plt.scatter(query_gt[0], query_gt[1], c='b', alpha=0.5)
        plt.scatter(pos_gt[0], pos_gt[1], c='g', marker='+', alpha=0.5)
        plt.scatter(neg_gt[:, 0], neg_gt[:, 1], c='r', marker='x', alpha=0.5)
        plt.savefig(os.path.join('train_batch_plot', str(query_index) + '_%d.png' % self.query_set.time_stamps[query_index]))
        plt.close()






