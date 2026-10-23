"""Class-label mapping: the deployed models' class order and the display names
shown in the UI, including the fault_3 -> F2 merge the thesis describes
(chapters/08_sensors.tex: "the fault_3 folder ... belong to F2").
"""

import json

import pytest

from app import inference

SHARED_CLASSES = ["fault_2", "fault_4", "fault_5", "fault_6", "fault_7", "fault_8", "fault_9", "fault_10", "fault_12", "fault_13"]


@pytest.mark.parametrize("sensor", ["1", "2", "3"])
def test_model_classes_are_exactly_the_ten_shared_classes(sensor):
    model = inference.get_model(sensor)
    assert model.classes == SHARED_CLASSES


@pytest.mark.parametrize("sensor", ["1", "2", "3"])
def test_fault_3_is_never_a_model_output(sensor):
    model = inference.get_model(sensor)
    assert "fault_3" not in model.classes
    assert "fault_5i" not in model.classes
    assert "fault_1" not in model.classes


def test_fault_2_display_name_notes_the_merge():
    assert inference.DISPLAY_NAMES["fault_2"] == "F2 (includes F3)"


def test_every_shared_class_has_a_display_name():
    for code in SHARED_CLASSES:
        assert code in inference.DISPLAY_NAMES
        assert inference.DISPLAY_NAMES[code].startswith("F")


@pytest.mark.parametrize("sensor", ["1", "2", "3"])
def test_resnet_keep_idx_matches_class_count(sensor):
    # meta.json's resnet_keep_idx selects which of the ResNet's raw outputs
    # correspond to `classes`; a length mismatch would silently misalign
    # predicted probabilities with class labels.
    meta_path = inference.CODE_ROOT / f"final_model_sensor{sensor}_10class" / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert len(meta["resnet_keep_idx"]) == len(meta["classes"]) == len(SHARED_CLASSES)
    assert meta["classes"] == SHARED_CLASSES
