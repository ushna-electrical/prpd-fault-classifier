"""Hyperparameters and dataset constants for the Stage 1 baseline classifier.

FAULT_CLASSES must match Stage 0's decided 13-class taxonomy exactly (see
prpd_data_prep/config.py and the plan's Dataset Audit) -- it is repeated here
rather than imported so this package has no hard dependency on prpd_data_prep,
but the two lists must be kept in sync.
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

# Fixed pixel crop of the raw 1800x1050 PRPD figure: isolates the plot's data
# region (heatmap + its axes box), discarding the matplotlib title (which bakes
# in filename/sensor/date as literal pixel text), axis tick labels, axis titles,
# and the colorbar -- per the plan's Dataset Audit (Finding 6). Verified against
# samples across multiple fault types and sensors: the axes spines sit at
# x in [116,126] (left) .. [1444,1447] (right), y in [55,56] (top) ..
# [962,963] (bottom); this box is a few pixels more generous on every side so
# no sample's plot content is clipped.
CROP_BOX = (100, 48, 1450, 968)  # (left, top, right, bottom)

BACKBONES = ("resnet50", "efficientnet_b0", "mobilenetv3_large")

DEFAULT_IMAGE_SIZE = 224
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 30
DEFAULT_LR = 3e-4
DEFAULT_PATIENCE = 5  # early stopping patience, on validation macro-F1
DEFAULT_SEED = 0

# Physically valid augmentation only (train split only). Deliberately NO flips
# and NO arbitrary rotation: the phase (x) axis is periodic 0-360 deg, so a
# flip/rotation would invert or scramble a physically meaningful signal.
#
# CircularPhaseShift: our own phase-periodicity augmentation.
MAX_PHASE_SHIFT_FRACTION = 0.15
#
# ElasticJitter / RandomZoom / RandomVerticalStretch: adopted from the PRPD
# dataset paper's (Research papers/10_PRPD-Dataset-Generators_DataBrief2023.md)
# validated augmentation set -- see transforms.py docstrings for the mapping
# from their Augmentor-library operations to these implementations, and for
# why their flip/shear operations are NOT adopted here.
ELASTIC_GRID_SIZE = (10, 7)
ELASTIC_MAX_DISPLACEMENT_FRACTION = 0.015
ZOOM_SCALE_RANGE = (0.95, 1.08)
VERTICAL_STRETCH_RANGE = (0.95, 1.08)
#
# Legacy augmentation, kept for use in earlier experiments (see
# stage1_output_mobilenetv3_large/ and stage2_output/), not used in the
# paper-#10-aligned default set above.
MAGNITUDE_JITTER_STRENGTH = 0.10
#
GAUSSIAN_NOISE_STD = 0.02

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
