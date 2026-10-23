"""Few-shot (Siamese network) PRPD fault classifier.

Implements the architecture from Tuyet-Doan et al., "One-Shot Learning for
Partial Discharge Diagnosis..." (Sensors 2020) -- see
Research papers/05_OneShotSiamese-GIS_Sensors2020.md -- adapted to a 13-way
N-shot setting on our own sensor-specific PRPD manifests, as the targeted
remedy for the classes that stayed at 0 F1 across every plain-softmax
MobileNetV3Large recipe tried in Stage 3 (fault_1, fault_3, fault_4, fault_5,
fault_5i -- exactly the smallest-support classes, not fixed by learning-rate
or backbone-freezing changes).
"""
