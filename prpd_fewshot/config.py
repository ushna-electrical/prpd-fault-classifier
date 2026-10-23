"""Hyperparameters for the Siamese few-shot classifier.

FAULT_CLASSES matches prpd_classifier/prpd_data_prep's decided 13-class
taxonomy -- repeated here for the same reason (no hard cross-package
dependency), must be kept in sync.
"""

FAULT_CLASSES = [
    "fault_1",
    "fault_2",
    "fault_3",
    "fault_4",
    "fault_5",
    "fault_5i",
    "fault_6",
    "fault_7",
    "fault_8",
    "fault_9",
    "fault_10",
    "fault_12",
    "fault_13",
]
NUM_CLASSES = len(FAULT_CLASSES)

# Same crop box as prpd_classifier (see that package's config.py docstring for
# how it was derived) -- isolates the plot's data region from the rendered
# title/axis/colorbar chrome.
CROP_BOX = (100, 48, 1450, 968)

# The Siamese branch is a small, shallow CNN (per the verified paper, not a
# large ImageNet backbone) -- appropriate for a genuinely small-sample method,
# and far cheaper to train on this CPU-only machine. A smaller input than the
# 224x224 used for MobileNetV3Large keeps this cheap without losing the
# coarse cluster-vs-floor structure the network needs to learn.
IMAGE_SIZE = 128
EMBEDDING_DIM = 64

DEFAULT_PAIRS_PER_EPOCH = 4000  # balanced same/different pairs, freshly resampled every epoch
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 20
DEFAULT_LR = 1e-3
DEFAULT_PATIENCE = 5  # early stopping on val N-shot macro-F1 (not pair accuracy)
DEFAULT_SEED = 0

DEFAULT_N_SHOT = 5  # support examples per class averaged into that class's prototype

# Prototypical-network episodic training (Snell et al. 2017) -- not one of the
# 20 reviewed papers, adopted because it directly matches the nearest-
# prototype objective used at eval time, unlike the pairwise verification loss.
DEFAULT_N_QUERY = 5  # query examples per class per training episode
DEFAULT_EPISODES_PER_EPOCH = 100

# Same physically-valid augmentation philosophy as prpd_classifier (no flips/
# rotation; the phase axis is periodic, the magnitude axis is not).
MAX_PHASE_SHIFT_FRACTION = 0.15
ELASTIC_GRID_SIZE = (10, 7)
ELASTIC_MAX_DISPLACEMENT_FRACTION = 0.015
ZOOM_SCALE_RANGE = (0.95, 1.08)
VERTICAL_STRETCH_RANGE = (0.95, 1.08)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
