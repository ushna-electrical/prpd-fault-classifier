"""Threshold logic: the three-tier confidence bands (app.inference.confidence_tier,
from the thesis's field-tool recommendation in chapters/09_calibration.tex) and the
voltage tie-break logic (test_model.FinalModel.voltage_check, used as-is by the
backend). The voltage tests use a real FinalModel so they exercise the actual
voltage_ranges.json data, not a copy of it.
"""

import pytest

from app import inference


# ---------------------------------------------------------------- confidence tiers

@pytest.mark.parametrize("confidence,expected_tier", [
    (1.0, "accept"),
    (0.9, "accept"),
    (0.89999, "review"),
    (0.75, "review"),
    (0.7, "review"),
    (0.69999, "manual"),
    (0.5, "manual"),
    (0.0, "manual"),
])
def test_confidence_tier_boundaries(confidence, expected_tier):
    assert inference.confidence_tier(confidence)["tier"] == expected_tier


def test_confidence_tier_labels_are_human_readable():
    assert "High" in inference.confidence_tier(0.95)["label"]
    assert "Review" in inference.confidence_tier(0.8)["label"]
    assert "manually" in inference.confidence_tier(0.5)["label"]


# ---------------------------------------------------------------- voltage tie-break

@pytest.fixture(scope="module")
def model():
    return inference.get_model("2")


def _result(top):
    """A minimal result dict shaped like FinalModel.predict()'s output, enough
    for voltage_check (which only reads result["top"])."""
    return {"top": top}


def test_not_tied_keeps_model_answer_regardless_of_voltage(model):
    # fault_7's pdiv range is roughly 3.8-10.3 kV; a voltage far outside it
    # must still not override an unambiguous top prediction.
    result = _result([("fault_7", 0.95), ("fault_2", 0.03), ("fault_9", 0.02)])
    check = model.voltage_check(result, voltage=20.0, margin=0.15)
    assert check["final"] == "fault_7"
    assert "not tied" in check["action"]


def test_tied_and_voltage_matches_one_overrides(model):
    # fault_5 (~11.7-18.5 kV) and fault_6 (~17.4-24.6 kV) tied; 12 kV is only
    # inside fault_5's range.
    result = _result([("fault_6", 0.50), ("fault_5", 0.49), ("fault_2", 0.01)])
    check = model.voltage_check(result, voltage=12.0, margin=0.15)
    assert check["tied"] == ["fault_6", "fault_5"]
    assert check["final"] == "fault_5"
    assert "overrides" in check["action"]


def test_tied_and_voltage_confirms_top_class(model):
    # 20 kV is inside fault_6's range but not fault_5's.
    result = _result([("fault_6", 0.50), ("fault_5", 0.49), ("fault_2", 0.01)])
    check = model.voltage_check(result, voltage=20.0, margin=0.15)
    assert check["final"] == "fault_6"
    assert "confirms" in check["action"]


def test_tied_and_voltage_matches_neither_keeps_model_answer(model):
    # 1 kV is below both fault_5's and fault_6's ranges.
    result = _result([("fault_6", 0.50), ("fault_5", 0.49), ("fault_2", 0.01)])
    check = model.voltage_check(result, voltage=1.0, margin=0.15)
    assert check["final"] == "fault_6"  # model's top answer kept
    assert "none of the tied classes" in check["action"]


def test_tie_margin_controls_what_counts_as_tied(model):
    result = _result([("fault_7", 0.60), ("fault_9", 0.45), ("fault_2", 0.01)])
    loose = model.voltage_check(result, voltage=5.0, margin=0.20)
    tight = model.voltage_check(result, voltage=5.0, margin=0.05)
    assert len(loose["tied"]) == 2  # 0.60 - 0.45 = 0.15 <= 0.20
    assert len(tight["tied"]) == 1  # 0.15 > 0.05
    assert "not tied" in tight["action"]
