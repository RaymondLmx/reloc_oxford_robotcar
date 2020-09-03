

class CONFIG:
    """
        config of the environment
    """
    # main
    MODE = 'train'
    RESUME = ''
    WHICH_CHECKPOINT = 'latest' # latest best
    DATASET_SIZE = 'whole'
    START_EPOCH = 0
    N_EPOCHS = 10
    N_PATIENCE = 10
    EVAL_EVERY_ = 1

    # path
    DATA_PATH = '../dataset'
    RUNS_PATH = 'runs'
    SAVE_PATH = 'checkpoints'
    CACHE_PATH = 'cache'

    # netvlad
    N_CLUSTERS = 64
    ENCODER_DIM = 512
    MARGIN = 0.1

    # optimizer
    LR = 0.001
    LR_STEP = 5
    LR_GAMMA = 0.5
    WEIGHT_DECAY = 0.001
    MOMENTUM = 0.9

    # loader
    BATCH_SIZE = 8
    N_WORKERS = 8

    # cache
    CACHE_BATCH_SIZE = 32
    CACHE_REFRESH_RATE = 1000

    # threads
    NONTRIV_POS_THREADS = 25
    POS_THREADS = 50

    # other
    SEED = 123
    IS_PARALLEL = False

