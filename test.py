import torch
import numpy as np
import dataset
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from config.template import CONFIG
from torch.utils.data import DataLoader, SubsetRandomSampler
from descriptor import descriptor
from os.path import join, exists, isfile
from os import makedirs
import faiss

# settings
MODE = 'test'
RESUME = ''
WHICH_CHECKPOINT = 'best'

cuda = torch.cuda.is_available()
device = torch.device('cuda' if cuda else 'cpu')
print(device)

result_path = join(RESUME, 'result')
if not exists(result_path):
    makedirs(result_path)

# ===================================================================================
# 1. load dataset
test_set = dataset.get_whole_test_set()
print('===> Evaluating on test set')
print('====> Query count:', test_set.query_set.num)

# ===================================================================================
# 2. build model
print('===> Building model')
model = descriptor(mode=MODE, resume=RESUME)

# ===================================================================================
# 3. checkpoint
if WHICH_CHECKPOINT == 'latest':
    checkpoint_path = join(RESUME, 'checkpoints', 'checkpoint.pth.tar')
elif WHICH_CHECKPOINT == 'best':
    checkpoint_path = join(RESUME, 'checkpoints', 'model_best.pth.tar')

if isfile(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    START_EPOCH = checkpoint['epoch']
    best_metric = checkpoint['best_score']
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    print("=> loaded checkpoint '{}' (epoch {})".format(checkpoint_path, checkpoint['epoch']))
else:
    print("=> no checkpoint found at '{}'".format(checkpoint_path))

# ===================================================================================
# 4. test
print('===> Running evaluation step')
epoch = 1
test_data_loader = DataLoader(dataset=test_set,
                              num_workers=CONFIG.N_WORKERS,
                              batch_size=CONFIG.BATCH_SIZE,
                              shuffle=False,
                              pin_memory=cuda)

model.eval()
with torch.no_grad():
    print('====> Extracting Features')
    pool_size = CONFIG.ENCODER_DIM * CONFIG.N_CLUSTERS
    sample_features = np.empty((len(test_set), pool_size))

    for i, (input, indices) in enumerate(test_data_loader, 1):
        input = input.to(device)
        image_encoding = model.encoder(input)
        vlad_encoding = model.pool(image_encoding)

        sample_features[indices.detach().numpy(), :] = vlad_encoding.detach().cpu().numpy()
        if i % 100 == 0 or len(test_data_loader) <= 10:
            print("==> Batch ({}/{})".format(i, len(test_data_loader)), flush=True)
        del input, image_encoding, vlad_encoding
del test_data_loader

query_features = sample_features[test_set.sample_set.num:].astype('float32')
sample_features = sample_features[: test_set.sample_set.num].astype('float32')

positives = test_set.get_positives()


# =====================================================================
# PR curve & f-score
knn = NearestNeighbors(n_jobs=-1)
knn.fit(sample_features)

dists, preds = knn.radius_neighbors(query_features, radius=2.0)

dist_threads = np.linspace(0.65, 1.4, num=127)

recalls = []
precisions = []
f1s = []
f2s = []
fbs = []

for dist_thread in dist_threads:
    r = []
    p = []

    for i, dist in enumerate(dists):
        idx = preds[i][np.where(dist < dist_thread)]

        if len(positives[i]):
            r.append(sum(np.in1d(idx, positives[i])) / len(positives[i]))

        if len(idx):
            p.append(np.mean(np.in1d(idx, positives[i])))

    recall = np.mean(r)
    precision = np.mean(p)
    f1 = 2 * precision * recall / (precision + recall)
    f2 = 5 * precision * recall / (4 * precision + recall)
    fb = 1.25 * precision * recall / (0.25 * precision + recall)

    recalls.append(recall)
    precisions.append(precision)
    f1s.append(f1)
    f2s.append(f2)
    fbs.append(fb)

# PR curve
plt.title('PR curve', fontsize=15)
plt.xlabel('recall', fontsize=14)
plt.ylabel('precision', fontsize=14)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.plot(recalls, precisions)
plt.savefig(join(result_path, 'pr_curve.png'))
plt.show()

# F-score
plt.title('F-score', fontsize=15)
plt.plot(range(len(f1s)), f1s, label='f1-score')
plt.plot(range(len(f2s)), f2s, label='f2-score')
plt.plot(range(len(fbs)), fbs, label='f0.5-score')
plt.legend()
plt.savefig(join(result_path, 'f_score.png'))
plt.show()
















