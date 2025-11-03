
"""
defaults.py

This module sets up the default configuration for training and evaluating 3D image models using the yacs library.
It includes model architecture parameters, training settings, data paths, optimizer configurations, and other
miscellaneous options.

Dependencies:
    - yacs.config: For hierarchical configuration management.

Owner:
    Edwing Ulin

Version:
    v1.0.0
"""
from yacs.config import CfgNode as CN

_C = CN()

# ---------------------------------------------------------------------------- #
# Model options
# ---------------------------------------------------------------------------- #
_C.MODEL = CN()

# Name of model
_C.MODEL.MODEL_NAME = "UNETDIP"

# Loss function, combined = dice loss + cross entropy, combined2 = dice loss + boundary loss
_C.MODEL.LOSS_FUNC = "l2"

_C.MODEL.L1_WEIGHT = 0.6
_C.MODEL.L2_WEIGHT = 0.4
_C.MODEL.SSIM_WEIGHT = 0

_C.MODEL.BLOCK_CONFIG = (4, 5, 7, 10, 12)
_C.MODEL.DROP_OUT_RATE = 0.2

# Filter dimensions for Input Interpolation block (currently all the same)
_C.MODEL.NUM_FILTERS_INTERPOL = 16
_C.MODEL.NUM_FILTERS = 15

# Number of input channels (slice thickness)
_C.MODEL.NUM_CHANNELS = 1

# Height of convolution kernels
_C.MODEL.KERNEL_H = 3

# Width of convolution kernels
_C.MODEL.KERNEL_W = 3

# Depth of convolution kernels
_C.MODEL.KERNEL_D = 3

# Stride during convolution
_C.MODEL.STRIDE_CONV = 1

_C.MODEL.STRIDE_POOL = 2

# Size of pooling filter
_C.MODEL.POOL = 2

_C.MODEL.N_CLASSES = 2

# Interpolation mode for up/downsampling in Flex networks
_C.MODEL.INTERPOLATION_MODE = "bilinear"

# ---------------------------------------------------------------------------- #
# Training options
# ---------------------------------------------------------------------------- #
_C.TRAIN = CN()

# input batch size for training
_C.TRAIN.BATCH_SIZE = 1

# how many batches to wait before logging training status
_C.TRAIN.LOG_INTERVAL = 20

_C.TRAIN.VAL_PLOT_INTERVAL = 10

# Resume training from the latest checkpoint in the output directory.
_C.TRAIN.RESUME = False

# The experiment number to resume from
_C.TRAIN.RESUME_EXPR_NUM = "Default"

# number of epochs to train
_C.TRAIN.NUM_EPOCHS = 1000

_C.TRAIN.NUM_WORKERS = 8



# ---------------------------------------------------------------------------- #
# Data options
# ---------------------------------------------------------------------------- #

_C.DATA = CN()

# path to data
_C.DATA.DATA_PATH = ""

# Available size for dataloader
# This for the multiscale dataloader
_C.DATA.SIZES = (134, 152, 152)


# ---------------------------------------------------------------------------- #
# DataLoader options (common for test and train)
# ---------------------------------------------------------------------------- #
_C.DATA_LOADER = CN()

# Number of data loader workers
_C.DATA_LOADER.NUM_WORKERS = 8

# Load data to pinned host memory.
_C.DATA_LOADER.PIN_MEMORY = True

# ---------------------------------------------------------------------------- #
# Optimizer options
# ---------------------------------------------------------------------------- #
_C.OPTIMIZER = CN()

# Base learning rate.
_C.OPTIMIZER.BASE_LR = 0.01

# Learning rate scheduler, step_lr, cosineWarmRestarts, reduceLROnPlateau
_C.OPTIMIZER.LR_SCHEDULER = "cosineWarmRestarts"

# Multiplicative factor of learning rate decay in step_lr
_C.OPTIMIZER.GAMMA = 0.3

# Period of learning rate decay in step_lr
_C.OPTIMIZER.STEP_SIZE = 5

# minimum learning in cosine lr policy and reduceLROnPlateau
_C.OPTIMIZER.ETA_MIN = 0.0001

# number of iterations for the first restart in cosineWarmRestarts
_C.OPTIMIZER.T_ZERO = 10

# A factor increases T_i after a restart in cosineWarmRestarts
_C.OPTIMIZER.T_MULT = 2

#MAXIMUM ITERATIONS
_C.OPTIMIZER.MAX_ITER = 300

#HISTORY SIZE
_C.OPTIMIZER.HISTORY_SIZE = 10

#WEIGHT DECAY
_C.OPTIMIZER.WEIGHT_DECAY = 1e-4

#MILESTONE
_C.OPTIMIZER.MILESTONE = [40]

# Patience for ReduceLROnPlateau scheduler
_C.OPTIMIZER.PATIENCE = 10

# Optimization method [lbfgs]
_C.OPTIMIZER.OPTIMIZING_METHOD = "lbfgs"

_C.OPTIMIZER.MOMENTUM = 0.9

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #

# Number of GPUs to use
_C.NUM_GPUS = 1

# log directory for run
_C.LOG_DIR = "./experiments"

# experiment number
_C.EXPR_NUM = "Default"

# Note that non-determinism may still be present due to non-deterministic
# operator implementations in GPU operator libraries.
_C.RNG_SEED = 1

_C.SUMMARY_PATH = "FastSurferVINN/summary/FastSurferVINN_coronal"

def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()