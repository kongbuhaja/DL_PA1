import os

# Training Hyperparameters
NUM_CLASSES                 = 200
BATCH_SIZE                  = 6128
VAL_EVERY_N_EPOCH           = 1

METRIC                      = 'f1score'

NUM_EPOCHS                  = 1000
# OPTIMIZER_PARAMS            = {'type': 'SGD', 'lr': 0.01, 'momentum': 0.9}
OPTIMIZER_PARAMS            = {'type': 'AdamW', 'lr': 0.01}
SCHEDULER_PARAMS            = {'type': 'MultiStepLR', 'milestones': [100, 200, 400, 600], 'gamma': 0.5}

# Dataset
DATASET_ROOT_PATH           = 'datasets/'
NUM_WORKERS                 = 8

# Augmentation
IMAGE_SIZE                  = 128
IMAGE_ROTATION              = 20
IMAGE_FLIP_PROB             = 0.5
IMAGE_NUM_CROPS             = 128
IMAGE_PAD_CROPS             = 4
IMAGE_MEAN                  = [0.4802, 0.4481, 0.3975]
IMAGE_STD                   = [0.2302, 0.2265, 0.2262]

# Network
MODEL_NAME                  = 'ViT'

# Compute related
ACCELERATOR                 = 'gpu'
DEVICES                     = [0]
PRECISION_STR               = '32-true'

# Logging
WANDB_PROJECT               = 'DL_PA1'
WANDB_ENTITY                = os.environ.get('WANDB_ENTITY')
WANDB_SAVE_DIR              = 'wandb/'
WANDB_IMG_LOG_FREQ          = 50
WANDB_NAME                  = f'{MODEL_NAME}-B{BATCH_SIZE}-{OPTIMIZER_PARAMS["type"]}'

if MODEL_NAME == 'ViT':
    PATCH_SIZE                  = 4
    HIDDEN_SIZE                 = 200
    MSA_DROPOUT_RATE            = 0
    HEAD_SIZE                   = 4
    MLP_SIZE                    = 500
    TRANSFORMER_SIZE            = 4
    TRANSFORMER_DROPOUT_RATE    = 0.1
    WANDB_NAME                 += f'-T{TRANSFORMER_SIZE}-P{PATCH_SIZE}-H{HIDDEN_SIZE}-M{MLP_SIZE}'

elif MODEL_NAME == 'AlexNet':
    CONV_SIZE                   = 128
    MLP_SIZE                    = 2048
    DROPOUT_RATE                = 0.4

# WANDB_NAME                  += f'-{SCHEDULER_PARAMS["type"]}{OPTIMIZER_PARAMS["lr"]:.1E}'
