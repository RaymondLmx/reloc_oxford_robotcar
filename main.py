import shutil
import torch
import torch.optim as optim
import torch.nn as nn
import h5py
import numpy as np
import dataset
import math
import json
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from datetime import datetime
from config.template import CONFIG
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.data.dataset import Subset
from descriptor import descriptor
from os import makedirs, remove
from os.path import join, exists, isfile
from sklearn.cluster import KMeans
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import faiss

# ===================================================================================
# settings
CFG = CONFIG()

cuda = torch.cuda.is_available()
device = torch.device('cuda' if cuda else 'cpu')

# ===================================================================================
def train(epoch):
    """
        train
    """
    epoch_loss = 0
    start_iter = 1

    if CONFIG.CACHE_REFRESH_RATE > 0:
        sub_set_num = math.ceil(len(train_set) / CONFIG.CACHE_REFRESH_RATE)
        sub_set_index = np.array_split(np.arange(len(train_set)), sub_set_num)
    else:
        sub_set_num = 1
        sub_set_index = [np.arange(len(train_set))]

    n_batch = (len(train_set) + CONFIG.BATCH_SIZE - 1) // CONFIG.BATCH_SIZE

    for sub_iter in range(sub_set_num):  # evaluate
        print('====> Building Cache')
        model.eval()
        train_set.cache = join(CONFIG.CACHE_PATH, train_set.query_set.dataset + '_feat_cache.hdf5')
        with h5py.File(train_set.cache, mode='w') as h5:
            pool_size = CONFIG.ENCODER_DIM * CONFIG.N_CLUSTERS
            h5_features = h5.create_dataset("features",
                                            [len(whole_train_set), pool_size],
                                            dtype=np.float32)
            with torch.no_grad():
                for i, (input, indices) in enumerate(whole_train_loader, 1):
                    input = input.to(device)
                    image_encoding = model.encoder(input)
                    vlad_encoding = model.pool(image_encoding)
                    h5_features[indices.detach().numpy(), :] = vlad_encoding.detach().cpu().numpy()
                    del input, image_encoding, vlad_encoding

        sub_train_set = Subset(dataset=train_set, indices=sub_set_index[sub_iter])

        train_data_loader = DataLoader(dataset=sub_train_set,
                                       num_workers=CONFIG.N_WORKERS,
                                       batch_size=CONFIG.BATCH_SIZE,
                                       shuffle=True,
                                       collate_fn=dataset.collate_fn,
                                       pin_memory=cuda)

        print('Allocated:', torch.cuda.memory_allocated())
        print('Cached:', torch.cuda.memory_cached())

        model.train()
        for iteration, (query, positive, negatives, neg_count, indices) in enumerate(train_data_loader, start_iter):

            if query is None:
                print('query is None')
                continue

            B, C, H, W = query.shape
            n_neg = torch.sum(neg_count)
            input = torch.cat([query, positive, negatives])

            input = input.to(device)
            image_encoding = model.encoder(input)
            vlad_encoding = model.pool(image_encoding)

            vladQ, vladP, vladN = torch.split(vlad_encoding, [B, B, n_neg])

            optimizer.zero_grad()

            # calculate loss for each Query, Positive, Negative triplet
            # due to potential difference in number of negatives have to
            # do it per query, per negative
            loss = 0
            for i, count in enumerate(neg_count):
                for c in range(count):
                    neg_index = (torch.sum(neg_count[:i]) + c).item()
                    loss += criterion(vladQ[i:i + 1], vladP[i:i + 1], vladN[neg_index: neg_index + 1])

            loss /= n_neg.float().to(device)
            loss.backward()
            optimizer.step()
            del input, image_encoding, vlad_encoding, vladQ, vladP, vladN
            del query, positive, negatives

            batch_loss = loss.item()
            epoch_loss += batch_loss

            if iteration % 10 == 0 or n_batch <= 10:
                print("==> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, iteration, n_batch, batch_loss), flush=True)
                writer.add_scalar('Train/Loss', batch_loss, ((epoch-1) * n_batch) + iteration)
                writer.add_scalar('Train/nNeg', n_neg, ((epoch-1) * n_batch) + iteration)
                # print('Allocated:', torch.cuda.memory_allocated())
                # print('Cached:', torch.cuda.memory_cached())

        start_iter += len(train_data_loader)
        del train_data_loader, loss
        optimizer.zero_grad()
        torch.cuda.empty_cache()
        remove(train_set.cache)

    avg_loss = epoch_loss / n_batch
    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, avg_loss), flush=True)
    writer.add_scalar('Train/AvgLoss', avg_loss, epoch)


def test(test_set, epoch=0, write_tboard=False):
    """
        test
    """
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
            if i % 50 == 0 or len(test_data_loader) <= 10:
                print("==> Batch ({}/{})".format(i, len(test_data_loader)), flush=True)
            del input, image_encoding, vlad_encoding
    del test_data_loader

    query_features = sample_features[test_set.n_samples:].astype('float32')
    sample_features = sample_features[: test_set.n_samples].astype('float32')

    print('====> Building faiss index')
    faiss_index = faiss.IndexFlatL2(pool_size)
    faiss_index.add(sample_features)

    print('====> Calculating recall @ N')
    n_values = [1, 5, 10, 20]

    _, predictions = faiss_index.search(query_features, max(n_values))

    gt = test_set.get_positives()

    correct_at_n = np.zeros(len(n_values))
    for q_index, pred in enumerate(predictions):
        for i, n in enumerate(n_values):
            if np.any(np.in1d(pred[:n], gt[q_index])):
                correct_at_n[i:] += 1
                break
    recall_at_n = correct_at_n / test_set.n_queries

    recalls = {}
    for i, n in enumerate(n_values):
        recalls[n] = recall_at_n[i]
        print("====> Recall@{}: {:.4f}".format(n, recall_at_n[i]))
        if write_tboard:
            writer.add_scalar('Val/Recall@' + str(n), recall_at_n[i], epoch)

    # plot the prediction
    _, pred = faiss_index.search(query_features, 1)
    pred_trajectory = test_set.samples_gt[pred.squeeze()]

    plt.title('trajectory', fontsize=15)
    plt.xlabel('x', fontsize=14)
    plt.ylabel('y', fontsize=14)
    plt.xlim(-200, 1200)
    plt.ylim(-500, 900)
    plt.scatter(pred_trajectory[:, 0], pred_trajectory[:, 1], s=0.5)
    plt.scatter(test_set.queries_gt[:, 0], test_set.queries_gt[:, 1], c='y', s=0.3)
    img_path = join(save_path, 'plt_cache', 'prediction.png')
    plt.savefig(img_path)
    plt.show()

    img_to_tensor = transforms.ToTensor()
    with Image.open(img_path) as img:
        writer.add_image('prediction', img_to_tensor(img), epoch)

    return recalls


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    model_out_path = join(save_path, filename)
    torch.save(state, model_out_path)
    if is_best:
        shutil.copyfile(model_out_path, join(save_path, 'model_best.pth.tar'))

# ===================================================================================
# 1. load dataset
print('===> Loading dataset(s)')
if CONFIG.MODE == 'train':
    whole_train_set = dataset.get_whole_train_set()
    whole_train_loader = DataLoader(dataset=whole_train_set,
                                    num_workers=CONFIG.N_WORKERS,
                                    batch_size=CONFIG.CACHE_BATCH_SIZE,
                                    shuffle=False,
                                    pin_memory=cuda)

    train_set = dataset.get_triplet_train_set()
    print('====> Training query set:', len(train_set))

    whole_val_set = dataset.get_whole_val_set()
    print('===> Evaluating on val set, query count:', whole_val_set.query_set.num)

elif CONFIG.MODE == 'test':
    whole_test_set = dataset.get_whole_test_set()
    print('===> Evaluating on test set')
    print('====> Query count:', whole_test_set.query_set.num)

# ===================================================================================
# 2. build model
print('===> Building model')
model = descriptor(mode=CONFIG.MODE, resume=CONFIG.RESUME)

if not CONFIG.RESUME:
    model = model.to(device)

# ===================================================================================
# 3. optimizer
optimizer = optim.SGD(filter(lambda p: p.requires_grad,  model.parameters()),
                      lr=CONFIG.LR,
                      momentum=CONFIG.MOMENTUM,
                      weight_decay=CONFIG.WEIGHT_DECAY)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=CONFIG.LR_STEP, gamma=CONFIG.LR_GAMMA)

# ===================================================================================
# 4. loss
criterion = nn.TripletMarginLoss(margin=CONFIG.MARGIN**0.5,
                                 p=2,
                                 reduction='sum').to(device)

# ===================================================================================
# 5. checkpoint TODO
if CONFIG.RESUME:
    if CONFIG.WHICH_CHECKPOINT == 'latest':
        checkpoint_path = join(CONFIG.RESUME, 'checkpoints', 'checkpoint.pth.tar')
    elif CONFIG.WHICH_CHECKPOINT == 'best':
        checkpoint_path = join(CONFIG.RESUME, 'checkpoints', 'model_best.pth.tar')

    if isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        CONFIG.START_EPOCH = checkpoint['epoch']
        best_metric = checkpoint['best_score']
        model.load_state_dict(checkpoint['state_dict'])
        model = model.to(device)
        if CONFIG.MODE == 'train':
            optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})".format(checkpoint_path, checkpoint['epoch']))

    else:
        print("=> no checkpoint found at '{}'".format(checkpoint_path))

# ===================================================================================
# 6. start
if CONFIG.MODE == 'test':
    print('===> Running evaluation step')
    epoch = 1
    recalls = test(whole_test_set, epoch)

elif CONFIG.MODE == 'train':
    print('===> Training model')

    writer = SummaryWriter(log_dir=join('runs', datetime.now().strftime('%b%d_%H-%M-%S') + '_' + 'vgg16' + '_' + 'netvlad'))

    # write checkpoints in logdir
    logdir = writer.file_writer.get_logdir()
    save_path = join(logdir, CONFIG.SAVE_PATH)
    if not CONFIG.RESUME:
        makedirs(save_path)

    with open(join(save_path, 'flag.json'), 'w') as f:
        f.write(json.dumps(
            {key: CONFIG.__dict__[key] for key in CONFIG.__dict__}))
    print('===> Saving state to:', logdir)

    not_imporved = 0
    best_score = 0
    for epoch in range(CONFIG.START_EPOCH, CONFIG.N_EPOCHS+1):

        scheduler.step(epoch)
        train(epoch)
        if (epoch % CONFIG.EVAL_EVERY_) == 0:
            recalls = test(whole_val_set, epoch, write_tboard=True)
            is_best = recalls[5] > best_score
            if is_best:
                not_imporved = 0
                best_score = recalls[5]
            else:
                not_imporved += 1

            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'recalls': recalls,
                'best_score': best_score,
                'optimizer': optimizer.state_dict(),
                'parallel': CONFIG.IS_PARALLEL,
            }, is_best)

            if CONFIG.N_PATIENCE > 0 and not_imporved > (CONFIG.N_PATIENCE / CONFIG.EVAL_EVERY_):
                print('Performance did not improve for', CONFIG.N_PATIENCE, 'epochs. Stopping.')
                break

    print("=> Best Recall@5: {:.4f}".format(best_score), flush=True)







