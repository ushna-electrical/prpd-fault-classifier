"""Hyperparameters reproducing Mück (2026), "KI-gestützte Auswertung von
Teilentladungsmustern von Energiekabelsystemen", sections 4.2-4.5 and 5.

Values marked (paper) are stated in the report; values marked (ours) are our
own choices where the paper's step cannot be reproduced from rendered PNGs.
"""

FAULT_CLASSES = [
    "fault_1", "fault_2", "fault_3", "fault_4", "fault_5", "fault_5i",
    "fault_6", "fault_7", "fault_8", "fault_9", "fault_10", "fault_12", "fault_13",
]

CROP_BOX = (135, 62, 1438, 955)  # strictly inside the axes (no spine pixels), as in prpd_scarcity
IMAGE_SIZE = 224  # (paper) ImageNet input size
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# colormap inversion / denoising (ours -- the paper masks the raw histogram)
BACKGROUND_MIN_CHANNEL = 250  # a pixel this bright in every channel is plot background
NOISE_ROW_OCCUPANCY = 0.90  # a charge row populated across >=90% of the phase axis is the noise floor
NOISE_ROW_STANDOUT = 1.15  # inside such a row, keep cells >15% brighter than the row's level

# training (paper, section 4.5 / 5.2.1)
DEFAULT_EPOCHS = 12
DEFAULT_BATCH_SIZE = 32
DEFAULT_LR = 5e-4  # cosine-annealed to LR_MIN
LR_MIN = 1e-6
WEIGHT_DECAY = 1e-2
DROPOUT = 0.2
HEAD_HIDDEN = 256
DEFAULT_SEED = 0

# layer-wise learning-rate scaling (paper, table 4.3)
LAYER_LR_SCALE = {"conv1": 0.001, "bn1": 0.001, "layer1": 0.001, "layer2": 0.01, "layer3": 0.1, "layer4": 0.5, "head": 1.0}

# (ours) The paper's factors leave conv1/layer1 at 5e-7, effectively frozen.
# That suited his near-lab-clean data; our own MobileNet experiments found heavy
# freezing badly harmful (macro-F1 0.128) and light freezing helpful, so this
# preset lets the backbone actually adapt to the PRPD domain.
LAYER_LR_SCALE_RELAXED = {"conv1": 0.1, "bn1": 0.1, "layer1": 0.1, "layer2": 0.2, "layer3": 0.5, "layer4": 1.0, "head": 1.0}
LR_SCALE_PRESETS = {"paper": LAYER_LR_SCALE, "relaxed": LAYER_LR_SCALE_RELAXED}

# augmentation probability range (paper: 80-100% per technique in iteration 1/2)
AUGMENT_PROB_MIN = 0.8
AUGMENT_PROB_MAX = 1.0
